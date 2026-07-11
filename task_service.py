# SPDX-License-Identifier: AGPL-3.0-or-later
"""Persistent, single-GPU task queue for agent-driven DrumLab jobs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

import numpy as np


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".aiff", ".opus"}
CH_NAMES = ["kick", "snare", "tom", "hihat", "cymbal"]
DEFAULT_THRESHOLDS = {"kick": 0.22, "snare": 0.24, "tom": 0.32, "hihat": 0.22, "cymbal": 0.30}
GRID_Q = {"1/8": Fraction(1, 2), "1/16": Fraction(1, 4), "1/16T": Fraction(1, 6), "1/32": Fraction(1, 8)}
GM_MAP = {"kick": 36, "snare": 38, "hihat": 42, "tom": 45, "cymbal": 49}
STAFF_MAP = {
    "kick": ("F", 4, "normal"),
    "snare": ("C", 5, "normal"),
    "tom": ("D", 5, "normal"),
    "hihat": ("G", 5, "x"),
    "cymbal": ("A", 5, "x"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w\-. ]+", "_", value)[:80] or "audio"


class TaskManager:
    """One durable queue, deliberately limited to one GPU pipeline at a time."""

    def __init__(self, app_dir: Path, python_executable: str):
        self.app_dir = Path(app_dir)
        self.python = python_executable
        self.root = self.app_dir / "workdir" / "tasks"
        self.root.mkdir(parents=True, exist_ok=True)
        self.worker_script = self.app_dir / "adtof_worker.py"
        self.lock = threading.RLock()
        self.tasks: dict[str, dict[str, Any]] = {}
        self.pending: "queue.Queue[str]" = queue.Queue()
        self.record_pending: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.port: Optional[int] = None
        self.allowed_roots: list[Path] = []
        self._load_existing()
        threading.Thread(target=self._worker_loop, name="drumlab-task-worker", daemon=True).start()
        threading.Thread(target=self._record_worker_loop, name="drumlab-record-worker", daemon=True).start()

    def configure(
        self,
        port: int,
        allowed_roots: Optional[list[str]] = None,
        output_root: Optional[str] = None,
    ) -> None:
        self.port = int(port)
        self.allowed_roots = []
        for value in allowed_roots or []:
            self.allowed_roots.append(Path(value).expanduser().resolve(strict=True))

        if output_root:
            target = Path(output_root).expanduser().resolve(strict=False)
            target.mkdir(parents=True, exist_ok=True)
            if not target.is_dir():
                raise ValueError(f"Task output root is not a directory: {target}")
            if target != self.root:
                # configure() runs before Uvicorn starts accepting requests, so it is
                # safe to switch the durable store and rebuild the in-memory index.
                with self.lock:
                    self.root = target
                    self.tasks.clear()
                    self._load_existing()

    def _load_existing(self) -> None:
        for path in self.root.glob("*/task.json"):
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
                if task.get("status") in ("queued", "running"):
                    task["status"] = "failed"
                    task["stage"] = "interrupted"
                    task["error"] = "Server restarted while the task was active"
                    task["updated_at"] = _now()
                    self._write(task)
                self.tasks[task["id"]] = task
            except (OSError, ValueError, KeyError):
                continue

    def _write(self, task: dict[str, Any]) -> None:
        folder = self.root / task["id"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "task.json"
        tmp = folder / f"task.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _public(self, task: dict[str, Any]) -> dict[str, Any]:
        data = json.loads(json.dumps(task))
        task_id = data["id"]
        data["urls"] = {
            "page": f"/tasks/{task_id}",
            "audio": f"/api/tasks/{task_id}/audio",
            "musicxml": f"/api/tasks/{task_id}/musicxml",
            "events": f"/api/tasks/{task_id}/events",
        }
        return data

    def _validate_port(self, service_port: int) -> None:
        if self.port is None:
            raise ValueError("Service port is not initialized yet")
        try:
            supplied = int(service_port)
        except (TypeError, ValueError):
            raise ValueError("service_port must be an integer") from None
        if supplied != self.port:
            raise ValueError(f"Port mismatch: this DrumLab service is running on {self.port}")

    def _validate_audio_path(self, audio_path: str) -> Path:
        if not audio_path or not Path(audio_path).is_absolute():
            raise ValueError("audio_path must be an absolute local path on the DrumLab server")
        try:
            path = Path(audio_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            raise ValueError("Audio file does not exist on the DrumLab server") from None
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            raise ValueError(f"Unsupported audio file; expected one of {sorted(AUDIO_EXTS)}")
        if self.allowed_roots and not any(path.is_relative_to(root) for root in self.allowed_roots):
            raise ValueError("audio_path is outside the configured --task-root directories")
        return path

    def submit(
        self,
        audio_path: str,
        service_port: int,
        *,
        model: str = "htdemucs",
        device: str = "cuda",
        source_mode: str = "full_mix",
        grid: str = "1/16",
        fps: int = 100,
        thresholds: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        self._validate_port(service_port)
        source = self._validate_audio_path(audio_path)
        if grid not in GRID_Q:
            raise ValueError(f"grid must be one of {list(GRID_Q)}")
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        if source_mode not in ("full_mix", "drum_only"):
            raise ValueError("source_mode must be full_mix or drum_only")
        if model not in ("htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"):
            raise ValueError("Unsupported Demucs model")
        if not 25 <= int(fps) <= 200:
            raise ValueError("fps must be between 25 and 200")
        merged_thresholds = dict(DEFAULT_THRESHOLDS)
        for key, value in (thresholds or {}).items():
            if key not in merged_thresholds:
                raise ValueError(f"Unknown drum class: {key}")
            merged_thresholds[key] = max(0.0, min(1.0, float(value)))

        task_id = uuid.uuid4().hex
        task = {
            "id": task_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "Waiting for the GPU worker",
            "error": None,
            "created_at": _now(),
            "updated_at": _now(),
            "source_path": str(source),
            "source_name": source.name,
            "duration": None,
            "tempo": None,
            "hit_counts": None,
            "options": {
                "model": model,
                "device": device,
                "source_mode": source_mode,
                "grid": grid,
                "fps": int(fps),
                "thresholds": merged_thresholds,
            },
            "recordings": {},
        }
        with self.lock:
            self.tasks[task_id] = task
            self._write(task)
        self.pending.put(task_id)
        return self._public(task)

    def get(self, task_id: str) -> dict[str, Any]:
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            return self._public(task)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        """Newest-first task snapshot for the demo/operations queue."""
        limit = max(1, min(500, int(limit)))
        with self.lock:
            ordered = sorted(
                self.tasks.values(),
                key=lambda item: item.get("created_at", ""),
                reverse=True,
            )
            return [self._public(task) for task in ordered[:limit]]

    def _update(self, task_id: str, **changes: Any) -> None:
        with self.lock:
            task = self.tasks[task_id]
            task.update(changes)
            task["updated_at"] = _now()
            self._write(task)

    def _log(self, task_id: str, message: str) -> None:
        folder = self.root / task_id
        with (folder / "pipeline.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[{_now()}] {message}\n")
        self._update(task_id, message=message)

    def _run(self, task_id: str, cmd: list[str], stage: str, progress: float) -> None:
        self._update(task_id, stage=stage, progress=progress)
        self._log(task_id, "Running: " + " ".join(cmd))
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            cmd,
            cwd=str(self.app_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            **kwargs,
        )
        (self.root / task_id / "pipeline.log").open("a", encoding="utf-8").write(result.stdout[-12000:])
        if result.returncode:
            raise RuntimeError(f"{stage} failed (exit {result.returncode}): {result.stdout[-800:]}")

    def _worker_loop(self) -> None:
        while True:
            task_id = self.pending.get()
            try:
                self._process(task_id)
            except Exception as exc:  # noqa: BLE001
                folder = self.root / task_id
                (folder / "pipeline.log").open("a", encoding="utf-8").write("\n" + traceback.format_exc())
                self._update(task_id, status="failed", stage="failed", error=str(exc), message=str(exc))
            finally:
                self.pending.task_done()

    def _process(self, task_id: str) -> None:
        task = self.tasks[task_id]
        folder = self.root / task_id
        opts = task["options"]
        self._update(task_id, status="running", stage="ingest", progress=0.02, error=None)

        input_wav = folder / "input.wav"
        self._run(
            task_id,
            ["ffmpeg", "-y", "-i", task["source_path"], "-vn", "-acodec", "pcm_s16le", str(input_wav)],
            "ingest",
            0.05,
        )

        if opts.get("source_mode", "full_mix") == "drum_only":
            # The input already contains drums only. Reuse the decoded WAV directly;
            # no Demucs model is loaded and no separation GPU time is consumed.
            drums_wav = input_wav
            self._update(task_id, stage="drum_input", progress=0.55)
            self._log(task_id, "Drum-only input selected; skipping Demucs separation")
        else:
            demucs_out = folder / "demucs"
            self._run(
                task_id,
                [
                    self.python, "-m", "demucs", "-n", opts["model"], "--shifts", "1",
                    "--overlap", "0.25", "-d", opts["device"], "-o", str(demucs_out), str(input_wav),
                ],
                "separation",
                0.12,
            )
            drums_found = next(demucs_out.rglob("drums.wav"), None)
            if drums_found is None:
                raise RuntimeError("Demucs completed without producing drums.wav")
            drums_wav = folder / "drums.wav"
            shutil.move(str(drums_found), drums_wav)
            shutil.rmtree(demucs_out, ignore_errors=True)

        acts_dir = folder / "activations"
        self._run(
            task_id,
            [
                self.python, str(self.worker_script), "--audio", str(drums_wav),
                "--out-dir", str(acts_dir), "--device", opts["device"], "--fps", str(opts["fps"]),
            ],
            "transcription",
            0.62,
        )

        self._update(task_id, stage="notation", progress=0.88)
        meta = json.loads((acts_dir / "meta.json").read_text(encoding="utf-8"))
        events = self._pick_events(acts_dir / "activations.npy", opts["thresholds"], opts["fps"])
        payload = {
            "task_id": task_id,
            "tempo": float(meta["tempo"]),
            "duration": float(meta["duration"]),
            "grid": opts["grid"],
            "thresholds": opts["thresholds"],
            "events": events,
            "counts": {name: len(values) for name, values in events.items()},
        }
        (folder / "events.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self._build_musicxml(events, float(meta["tempo"]), opts["grid"], folder / "score.musicxml")
        self._build_midi(events, float(meta["tempo"]), folder / "performance.mid")

        self._update(
            task_id,
            status="completed",
            stage="completed",
            progress=1.0,
            message="Dynamic drum score is ready",
            duration=float(meta["duration"]),
            tempo=float(meta["tempo"]),
            hit_counts=payload["counts"],
        )

    def _post_processing(self):
        spec = importlib.util.find_spec("adtof_pytorch")
        if not spec or not spec.origin:
            raise RuntimeError("adtof_pytorch is not installed in the DrumLab environment")
        path = Path(spec.origin).parent / "post_processing.py"
        mod_spec = importlib.util.spec_from_file_location("drumlab_task_post_processing", path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        return module

    def _pick_events(self, activation_path: Path, thresholds: dict[str, float], fps: int) -> dict[str, list[float]]:
        arr = np.load(str(activation_path))
        ordered = [float(thresholds[name]) for name in CH_NAMES]
        picker = self._post_processing().PeakPicker(thresholds=ordered, fps=int(fps))
        picked = picker.pick(arr, labels=list(range(5)), label_offset=0)[0]
        return {CH_NAMES[i]: [round(max(0.0, float(t)), 5) for t in picked[i]] for i in range(5)}

    @staticmethod
    def _quantize(events: dict[str, list[float]], tempo: float, grid: str) -> dict[str, list[tuple[int, float]]]:
        frac = GRID_Q[grid]
        step_sec = float(frac) * 60.0 / tempo
        result: dict[str, list[tuple[int, float]]] = {}
        for name, times in events.items():
            slots = sorted({round(value / step_sec) for value in times})
            result[name] = [(slot, slot * step_sec) for slot in slots]
        return result

    def _build_musicxml(self, events: dict[str, list[float]], tempo: float, grid: str, path: Path) -> None:
        from music21 import clef, duration as m21dur, meter, note, percussion, stream, tempo as m21tempo

        frac = GRID_Q[grid]
        by_offset: dict[int, set[str]] = {}
        for name, rows in self._quantize(events, tempo, grid).items():
            for slot, _ in rows:
                by_offset.setdefault(slot, set()).add(name)

        def unpitched(name: str):
            step, octave, head = STAFF_MAP[name]
            value = note.Unpitched()
            value.displayStep = step
            value.displayOctave = octave
            if head != "normal":
                value.notehead = head
            return value

        part = stream.Part()
        part.partName = "Drums"
        part.insert(0, clef.PercussionClef())
        part.insert(0, meter.TimeSignature("4/4"))
        part.insert(0, m21tempo.MetronomeMark(number=round(tempo, 2)))
        for slot in sorted(by_offset):
            names = sorted(by_offset[slot])
            element = unpitched(names[0]) if len(names) == 1 else percussion.PercussionChord([unpitched(n) for n in names])
            element.duration = m21dur.Duration(frac)
            part.insert(Fraction(slot) * frac, element)
        stream.Score([part]).write("musicxml", fp=str(path))

    @staticmethod
    def _build_midi(events: dict[str, list[float]], tempo: float, path: Path) -> None:
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0, is_drum=True, name="ADTOF drums")
        for name, times in events.items():
            for value in times:
                instrument.notes.append(pretty_midi.Note(velocity=100, pitch=GM_MAP[name], start=value, end=value + 0.1))
        midi.instruments.append(instrument)
        midi.write(str(path))

    def artifact(self, task_id: str, name: str) -> Path:
        task = self.get(task_id)
        if task["status"] != "completed":
            raise RuntimeError("Task is not completed")
        allowed = {"audio": "input.wav", "musicxml": "score.musicxml", "events": "events.json", "midi": "performance.mid"}
        if name not in allowed:
            raise KeyError(name)
        path = self.root / task_id / allowed[name]
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _video_dimensions(aspect_ratio: str, width: int, height: Optional[int]) -> tuple[int, int]:
        width = max(320, min(3840, int(width)))
        if height is None:
            try:
                left, right = aspect_ratio.split(":", 1)
                ratio = float(left) / float(right)
                height = round(width / ratio)
            except (ValueError, ZeroDivisionError):
                raise ValueError("aspect_ratio must look like 16:9, 9:16, 4:3, or 1:1") from None
        height = max(240, min(3840, int(height)))
        return width // 2 * 2, height // 2 * 2

    def create_recording(
        self,
        task_id: str,
        service_port: int,
        *,
        aspect_ratio: str = "16:9",
        width: int = 1920,
        height: Optional[int] = None,
    ) -> dict[str, Any]:
        self._validate_port(service_port)
        task = self.get(task_id)
        if task["status"] != "completed":
            raise ValueError("The score task must be completed before recording")
        width, height = self._video_dimensions(aspect_ratio, width, height)
        recording_id = uuid.uuid4().hex
        recording = {
            "id": recording_id,
            "status": "queued",
            "width": width,
            "height": height,
            "created_at": _now(),
            "updated_at": _now(),
            "error": None,
            "url": f"/api/tasks/{task_id}/recordings/{recording_id}/video",
        }
        with self.lock:
            self.tasks[task_id]["recordings"][recording_id] = recording
            self._write(self.tasks[task_id])
        self.record_pending.put((task_id, recording_id))
        return recording

    def get_recording(self, task_id: str, recording_id: str) -> dict[str, Any]:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task or recording_id not in task.get("recordings", {}):
                raise KeyError(recording_id)
            return json.loads(json.dumps(task["recordings"][recording_id]))

    def _record_update(self, task_id: str, recording_id: str, **changes: Any) -> None:
        with self.lock:
            recording = self.tasks[task_id]["recordings"][recording_id]
            recording.update(changes)
            recording["updated_at"] = _now()
            self._write(self.tasks[task_id])

    def _record_worker_loop(self) -> None:
        while True:
            task_id, recording_id = self.record_pending.get()
            try:
                self._record(task_id, recording_id)
            except Exception as exc:  # noqa: BLE001
                self._record_update(task_id, recording_id, status="failed", error=str(exc))
            finally:
                self.record_pending.task_done()

    def _record(self, task_id: str, recording_id: str) -> None:
        from playwright.sync_api import sync_playwright

        recording = self.get_recording(task_id, recording_id)
        folder = self.root / task_id / "recordings" / recording_id
        folder.mkdir(parents=True, exist_ok=True)
        self._record_update(task_id, recording_id, status="running")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(
                viewport={"width": recording["width"], "height": recording["height"]},
                record_video_dir=str(folder),
                record_video_size={"width": recording["width"], "height": recording["height"]},
            )
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{self.port}/tasks/{task_id}?recording=1", wait_until="networkidle")
            page.wait_for_function("window.__DRUMLAB_SCORE_READY__ === true", timeout=120000)
            page.click("#play")
            duration_ms = int((float(self.tasks[task_id]["duration"] or 0) + 1.5) * 1000)
            page.wait_for_timeout(duration_ms)
            video = page.video
            context.close()
            source = Path(video.path())
            final = folder / "dynamic-score.webm"
            shutil.move(str(source), final)
            browser.close()
        self._record_update(task_id, recording_id, status="completed", path=str(final))

    def recording_file(self, task_id: str, recording_id: str) -> Path:
        recording = self.get_recording(task_id, recording_id)
        if recording["status"] != "completed":
            raise RuntimeError("Recording is not completed")
        path = Path(recording["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        return path


TASK_MANAGER: Optional[TaskManager] = None


def get_task_manager(app_dir: Path, python_executable: str = sys.executable) -> TaskManager:
    global TASK_MANAGER
    if TASK_MANAGER is None:
        TASK_MANAGER = TaskManager(app_dir, python_executable)
    return TASK_MANAGER
