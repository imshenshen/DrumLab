# SPDX-License-Identifier: AGPL-3.0-or-later
"""Persistent, single-GPU task queue for agent-driven DrumLab jobs."""

from __future__ import annotations

import hashlib
import bisect
import importlib.util
import json
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import math
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
SCORE_VERSION = 9  # v9 quantizes against a persisted per-beat timing map


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w\-. ]+", "_", value)[:80] or "audio"


class TaskCancelled(RuntimeError):
    """Raised inside the queue worker after a user-requested stop."""


class TaskTimedOut(RuntimeError):
    """Raised when a task exceeds the configured whole-pipeline deadline."""


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
        self.task_timeout_seconds = 3600.0
        self.recording_browser_executable: Optional[str] = None
        self.cancel_requested: set[str] = set()
        self.active_processes: dict[str, subprocess.Popen] = {}
        self.deadlines: dict[str, Optional[float]] = {}
        self._load_existing()
        threading.Thread(target=self._worker_loop, name="drumlab-task-worker", daemon=True).start()
        threading.Thread(target=self._record_worker_loop, name="drumlab-record-worker", daemon=True).start()

    def configure(
        self,
        port: int,
        allowed_roots: Optional[list[str]] = None,
        output_root: Optional[str] = None,
        task_timeout_seconds: float = 3600,
        recording_browser: Optional[str] = None,
    ) -> None:
        self.port = int(port)
        try:
            timeout = float(task_timeout_seconds)
        except (TypeError, ValueError):
            raise ValueError("Task timeout must be a number of seconds") from None
        if timeout < 0:
            raise ValueError("Task timeout cannot be negative; use 0 to disable it")
        self.task_timeout_seconds = timeout
        configured_browser = recording_browser or os.environ.get("DRUMLAB_RECORDING_BROWSER")
        if configured_browser:
            browser_path = Path(configured_browser).expanduser().resolve(strict=True)
            if not browser_path.is_file():
                raise ValueError(f"Recording browser is not a file: {browser_path}")
            self.recording_browser_executable = str(browser_path)
        else:
            self.recording_browser_executable = next((
                path for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")
                if (path := shutil.which(name))
            ), None)
        self.allowed_roots = []
        for value in allowed_roots or []:
            self.allowed_roots.append(Path(value).expanduser().resolve(strict=True))

        if output_root:
            target = Path(output_root).expanduser().resolve(strict=False)
            target.mkdir(parents=True, exist_ok=True)
            if not target.is_dir():
                raise ValueError(f"Task output root is not a directory: {target}")
            self._verify_output_root(target)
            if target != self.root:
                # configure() runs before Uvicorn starts accepting requests, so it is
                # safe to switch the durable store and rebuild the in-memory index.
                with self.lock:
                    self.root = target
                    self.tasks.clear()
                    self._load_existing()

    @staticmethod
    def _verify_output_root(root: Path) -> None:
        """Prove the service can create task directories, not merely see the mount."""
        probe = root / f".drumlab-write-test-{uuid.uuid4().hex}"
        marker = probe / "write-test"
        try:
            probe.mkdir()
            marker.write_text("ok", encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"Task output root is not writable by the DrumLab process: {root} ({exc})"
            ) from exc
        finally:
            try:
                marker.unlink(missing_ok=True)
                probe.rmdir()
            except OSError:
                pass

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
        beats_per_measure: int = 4,
        beat_unit: int = 4,
        measures_per_system: int = 3,
        pickup_mode: str = "auto",
        pickup_beats: float = 0.0,
        notation_offset_seconds: Optional[float] = None,
        notation_tempo: Optional[float] = None,
        timing_mode: str = "beat_map",
        fps: int = 100,
        thresholds: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        self._validate_port(service_port)
        source = self._validate_audio_path(audio_path)
        if grid not in GRID_Q:
            raise ValueError(f"grid must be one of {list(GRID_Q)}")
        beats_per_measure = int(beats_per_measure)
        beat_unit = int(beat_unit)
        if not 1 <= beats_per_measure <= 32:
            raise ValueError("beats_per_measure must be between 1 and 32")
        if beat_unit not in (1, 2, 4, 8, 16, 32):
            raise ValueError("beat_unit must be one of 1, 2, 4, 8, 16, or 32")
        measure_quarter_length = Fraction(beats_per_measure * 4, beat_unit)
        if (measure_quarter_length / GRID_Q[grid]).denominator != 1:
            raise ValueError("grid must divide the selected time signature into whole slots")
        measures_per_system = int(measures_per_system)
        if not 1 <= measures_per_system <= 6:
            raise ValueError("measures_per_system must be between 1 and 6")
        if pickup_mode not in ("none", "manual", "auto"):
            raise ValueError("pickup_mode must be none, manual, or auto")
        pickup_beats = float(pickup_beats)
        if pickup_beats < 0 or pickup_beats >= beats_per_measure:
            raise ValueError("pickup_beats must be at least 0 and less than beats_per_measure")
        if notation_offset_seconds is not None:
            notation_offset_seconds = float(notation_offset_seconds)
            if notation_offset_seconds < 0:
                raise ValueError("notation_offset_seconds cannot be negative")
        if notation_tempo is not None:
            notation_tempo = float(notation_tempo)
            if not 20 <= notation_tempo <= 999:
                raise ValueError("notation_tempo must be between 20 and 999 BPM")
        if timing_mode not in ("beat_map", "fixed"):
            raise ValueError("timing_mode must be beat_map or fixed")
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
            "timeout_seconds": self.task_timeout_seconds,
            "options": {
                "model": model,
                "device": device,
                "source_mode": source_mode,
                "grid": grid,
                "beats_per_measure": beats_per_measure,
                "beat_unit": beat_unit,
                "measures_per_system": measures_per_system,
                "pickup_mode": pickup_mode,
                "pickup_beats": pickup_beats if pickup_mode == "manual" else 0.0,
                "pickup_confidence": None,
                "notation_offset_seconds": notation_offset_seconds,
                "notation_tempo": notation_tempo,
                "timing_mode": timing_mode,
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
            needs_upgrade = (
                task.get("status") == "completed"
                and int(task.get("score_version") or 0) < SCORE_VERSION
            )
        if needs_upgrade:
            self._upgrade_musicxml(task_id)
        with self.lock:
            return self._public(self.tasks[task_id])

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

    def stop(self, task_id: str) -> dict[str, Any]:
        """Cancel a queued task or terminate the active task's whole process group."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.get("status") in ("completed", "failed", "cancelled", "timed_out"):
                return self._public(task)
            self.cancel_requested.add(task_id)
            process = self.active_processes.get(task_id)
            if task.get("status") == "queued":
                task.update({
                    "status": "cancelled",
                    "stage": "cancelled",
                    "message": "Task cancelled before it started",
                    "error": None,
                    "updated_at": _now(),
                })
            else:
                task.update({
                    "stage": "stopping",
                    "message": "Stopping task and child processes ...",
                    "updated_at": _now(),
                })
            self._write(task)
        if process is not None:
            self._terminate_process_tree(process)
        return self.get(task_id)

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

    def _check_abort(self, task_id: str) -> None:
        with self.lock:
            if task_id in self.cancel_requested:
                raise TaskCancelled("Task stopped by user")
            deadline = self.deadlines.get(task_id)
        if deadline is not None and time.monotonic() >= deadline:
            raise TaskTimedOut("Task exceeded its configured timeout")

    def _remaining_timeout(self, task_id: str) -> Optional[float]:
        with self.lock:
            deadline = self.deadlines.get(task_id)
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TaskTimedOut("Task exceeded its configured timeout")
        return remaining

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _run(self, task_id: str, cmd: list[str], stage: str, progress: float) -> None:
        self._check_abort(task_id)
        self._update(task_id, stage=stage, progress=progress)
        self._log(task_id, "Running: " + " ".join(cmd))
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(
            cmd,
            cwd=str(self.app_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            **kwargs,
        )
        with self.lock:
            self.active_processes[task_id] = process
        output = ""
        try:
            try:
                # Close the race where stop() lands after Popen but before the
                # process is registered in active_processes.
                self._check_abort(task_id)
                output, _ = process.communicate(timeout=self._remaining_timeout(task_id))
            except subprocess.TimeoutExpired:
                self._terminate_process_tree(process)
                output, _ = process.communicate()
                raise TaskTimedOut("Task exceeded its configured timeout") from None
            except (TaskCancelled, TaskTimedOut):
                self._terminate_process_tree(process)
                output, _ = process.communicate()
                raise
        finally:
            with self.lock:
                if self.active_processes.get(task_id) is process:
                    self.active_processes.pop(task_id, None)
        with (self.root / task_id / "pipeline.log").open("a", encoding="utf-8") as handle:
            handle.write((output or "")[-12000:])
        self._check_abort(task_id)
        if process.returncode:
            raise RuntimeError(f"{stage} failed (exit {process.returncode}): {(output or '')[-800:]}")

    def _worker_loop(self) -> None:
        while True:
            task_id = self.pending.get()
            try:
                if self.tasks.get(task_id, {}).get("status") == "cancelled":
                    continue
                self._process(task_id)
            except TaskCancelled as exc:
                self._update(task_id, status="cancelled", stage="cancelled", error=None,
                             message=str(exc))
            except TaskTimedOut as exc:
                self._update(task_id, status="timed_out", stage="timed_out", error=str(exc),
                             message=str(exc))
            except Exception as exc:  # noqa: BLE001
                folder = self.root / task_id
                (folder / "pipeline.log").open("a", encoding="utf-8").write("\n" + traceback.format_exc())
                self._update(task_id, status="failed", stage="failed", error=str(exc), message=str(exc))
            finally:
                with self.lock:
                    self.active_processes.pop(task_id, None)
                    self.deadlines.pop(task_id, None)
                    self.cancel_requested.discard(task_id)
                self.pending.task_done()

    def _process(self, task_id: str) -> None:
        task = self.tasks[task_id]
        folder = self.root / task_id
        opts = task["options"]
        timeout = float(task.get("timeout_seconds", self.task_timeout_seconds))
        with self.lock:
            self.deadlines[task_id] = time.monotonic() + timeout if timeout > 0 else None
        self._check_abort(task_id)
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
        self._check_abort(task_id)
        meta = json.loads((acts_dir / "meta.json").read_text(encoding="utf-8"))
        events = self._pick_events(acts_dir / "activations.npy", opts["thresholds"], opts["fps"])
        notation_offset_seconds = opts.get("notation_offset_seconds")
        notation_tempo = opts.get("notation_tempo")
        if notation_tempo is None:
            notation_tempo = self._refine_notation_tempo(events, float(meta["tempo"]), opts["grid"])
            opts["notation_tempo_mode"] = "auto"
        else:
            notation_tempo = float(notation_tempo)
            opts["notation_tempo_mode"] = "manual"
        opts["notation_tempo"] = notation_tempo
        if notation_offset_seconds is None:
            notation_offset_seconds = self._default_notation_offset(events)
        opts["notation_offset_seconds"] = notation_offset_seconds
        raw_beat_times = [float(value) for value in meta.get("beat_times", [])]
        timing_mode = opts.get("timing_mode", "beat_map")
        if timing_mode == "beat_map" and len(raw_beat_times) < 2:
            timing_mode = "fixed"
        opts["timing_mode"] = timing_mode
        score_beat_times = self._align_beat_times(
            raw_beat_times, notation_offset_seconds, float(meta["duration"])
        ) if timing_mode == "beat_map" else []
        pickup_beats, pickup_confidence = self._resolve_pickup(
            events,
            notation_tempo,
            opts["grid"],
            opts.get("beats_per_measure", 4),
            opts.get("beat_unit", 4),
            opts.get("pickup_mode", "auto"),
            opts.get("pickup_beats", 0.0),
            score_beat_times,
        )
        opts["pickup_beats"] = pickup_beats
        opts["pickup_confidence"] = pickup_confidence
        payload = {
            "task_id": task_id,
            "tempo": float(meta["tempo"]),
            "duration": float(meta["duration"]),
            "grid": opts["grid"],
            "beats_per_measure": opts.get("beats_per_measure", 4),
            "beat_unit": opts.get("beat_unit", 4),
            "measures_per_system": opts.get("measures_per_system", 3),
            "pickup_mode": opts.get("pickup_mode", "auto"),
            "pickup_beats": pickup_beats,
            "pickup_confidence": pickup_confidence,
            "notation_offset_seconds": notation_offset_seconds,
            "notation_tempo": notation_tempo,
            "notation_tempo_mode": opts["notation_tempo_mode"],
            "timing_mode": timing_mode,
            "beat_times": raw_beat_times,
            "score_beat_times": score_beat_times,
            "thresholds": opts["thresholds"],
            "events": events,
            "counts": {name: len(values) for name, values in events.items()},
        }
        (folder / "events.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self._build_musicxml(
            events,
            notation_tempo,
            opts["grid"],
            folder / "score.musicxml",
            duration_seconds=float(meta["duration"]),
            beats_per_measure=opts.get("beats_per_measure", 4),
            beat_unit=opts.get("beat_unit", 4),
            measures_per_system=opts.get("measures_per_system", 3),
            pickup_beats=pickup_beats,
            notation_offset_seconds=notation_offset_seconds,
            beat_times=score_beat_times,
            timing_mode=timing_mode,
        )
        self._build_midi(events, float(meta["tempo"]), folder / "performance.mid")
        self._check_abort(task_id)

        self._update(
            task_id,
            status="completed",
            stage="completed",
            progress=1.0,
            message="Dynamic drum score is ready",
            duration=float(meta["duration"]),
            tempo=float(meta["tempo"]),
            hit_counts=payload["counts"],
            score_version=SCORE_VERSION,
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

    @staticmethod
    def _pickup_slots(pickup_beats: float, beats_per_measure: int, slots_per_measure: int) -> tuple[int, float]:
        slots = round(float(pickup_beats) * slots_per_measure / beats_per_measure)
        slots = max(0, min(slots_per_measure - 1, slots))
        effective_beats = slots * beats_per_measure / slots_per_measure
        return slots, round(effective_beats, 6)

    @staticmethod
    def _default_notation_offset(events: dict[str, list[float]]) -> float:
        times = [float(value) for values in events.values() for value in values]
        return round(min(times, default=0.0), 6)

    @staticmethod
    def _refine_notation_tempo(events: dict[str, list[float]], seed_tempo: float, grid: str) -> float:
        """Fine-fit a global score BPM around the detector seed to limit long-song drift."""
        times = sorted(float(value) for values in events.values() for value in values)
        if len(times) < 8 or times[-1] - times[0] < 8:
            return round(float(seed_tempo), 6)
        origin = times[0]
        relative = [value - origin for value in times]
        if len(relative) > 2000:
            stride = max(1, len(relative) // 2000)
            relative = relative[::stride]

        grid_quarters = float(GRID_Q[grid])

        def loss(candidate: float) -> float:
            step_seconds = grid_quarters * 60.0 / candidate
            total = 0.0
            for value in relative:
                position = value / step_seconds
                error = abs(position - round(position))
                total += min(error, 0.30) ** 2
            # Keep the fit local when two nearly equivalent grids exist.
            total /= len(relative)
            total += abs(candidate - seed_tempo) / seed_tempo * 0.0002
            return total

        span = float(seed_tempo) * 0.01
        coarse_step = span * 2 / 400
        coarse = [float(seed_tempo) - span + index * coarse_step for index in range(401)]
        best = min(coarse, key=loss)
        fine_step = coarse_step / 100
        fine = [best - coarse_step + index * fine_step for index in range(201)]
        return round(min(fine, key=loss), 6)

    @staticmethod
    def _align_beat_times(beat_times: list[float], origin: float, duration: float) -> list[float]:
        values = sorted(float(value) for value in beat_times if float(value) >= 0)
        if len(values) < 2:
            return []
        anchor = min(range(len(values)), key=lambda index: abs(values[index] - origin))
        shift = float(origin) - values[anchor]
        aligned = [value + shift for value in values[anchor:]]
        aligned[0] = float(origin)
        intervals = [b - a for a, b in zip(aligned, aligned[1:]) if b > a]
        fallback = sorted(intervals)[len(intervals) // 2] if intervals else 0.5
        while aligned[-1] < duration + fallback:
            interval = intervals[-1] if intervals else fallback
            aligned.append(aligned[-1] + interval)
        return [round(value, 6) for value in aligned]

    def _detect_beat_times_for_existing_task(self, task_id: str, payload: dict[str, Any]) -> list[float]:
        existing = [float(value) for value in payload.get("beat_times", [])]
        if len(existing) >= 2:
            return existing
        folder = self.root / task_id
        acts_dir = folder / "activations"
        acts_dir.mkdir(parents=True, exist_ok=True)
        meta_path = acts_dir / "meta.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps({
                "tempo": payload.get("tempo"), "duration": payload.get("duration")
            }), encoding="utf-8")
        kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 300}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run([
                self.python, str(self.worker_script), "--audio", str(folder / "input.wav"),
                "--out-dir", str(acts_dir), "--tempo-only",
            ], **kwargs)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return []
        if result.returncode != 0:
            return []
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return [float(value) for value in meta.get("beat_times", [])]
        except (OSError, ValueError, TypeError):
            return []

    def _resolve_pickup(
        self,
        events: dict[str, list[float]],
        tempo: float,
        grid: str,
        beats_per_measure: int,
        beat_unit: int,
        pickup_mode: str,
        pickup_beats: float,
        beat_times: Optional[list[float]] = None,
    ) -> tuple[float, Optional[float]]:
        measure_quarter_length = Fraction(beats_per_measure * 4, beat_unit)
        slots_per_measure = int(measure_quarter_length / GRID_Q[grid])
        if pickup_mode == "none":
            return 0.0, None
        if pickup_mode == "manual":
            _, effective = self._pickup_slots(pickup_beats, beats_per_measure, slots_per_measure)
            return effective, None

        mapped_beats = [float(value) for value in (beat_times or [])]
        slots_per_beat = max(1, round(slots_per_measure / beats_per_measure))
        if len(mapped_beats) >= 2:
            slots_by_class = {}
            for name, values in events.items():
                mapped = []
                for value in values:
                    value = float(value)
                    beat_index = bisect.bisect_right(mapped_beats, value) - 1
                    if beat_index < 0:
                        continue
                    beat_index = min(beat_index, len(mapped_beats) - 2)
                    interval = max(1e-6, mapped_beats[beat_index + 1] - mapped_beats[beat_index])
                    subdivision = round((value - mapped_beats[beat_index]) / interval * slots_per_beat)
                    mapped.append(beat_index * slots_per_beat + subdivision)
                slots_by_class[name] = mapped
        else:
            raw_times = [float(value) for values in events.values() for value in values]
            first_time = min(raw_times, default=0.0)
            step_seconds = float(GRID_Q[grid]) * 60.0 / tempo
            slots_by_class = {
                name: [round((float(value) - first_time) / step_seconds) for value in values]
                for name, values in events.items()
            }
        all_slots = [slot for slots in slots_by_class.values() for slot in slots]
        if len(all_slots) < 8:
            return 0.0, 0.0

        # Score each possible bar phase. Kick/cymbal accents are strongest on a
        # downbeat; in 4/4, snare backbeats add useful evidence. This is deliberately
        # conservative: ambiguous material remains non-pickup and can be corrected in UI.
        downbeat_weights = {"kick": 4.0, "cymbal": 3.0, "tom": 1.2, "snare": 0.5, "hihat": 0.2}
        scores: list[tuple[float, int]] = []
        for candidate in range(slots_per_measure):
            if candidate and not any(slot < candidate for slot in all_slots):
                continue
            score = 0.0
            for name, slots in slots_by_class.items():
                for slot in slots:
                    position = (slot - candidate) % slots_per_measure
                    if position == 0:
                        score += downbeat_weights[name]
                    if beats_per_measure == 4 and name == "snare":
                        if position in (slots_per_measure // 4, slots_per_measure * 3 // 4):
                            score += 2.2
                    if beats_per_measure == 4 and name == "kick" and position == slots_per_measure // 2:
                        score += 1.0
            scores.append((score, candidate))
        scores.sort(reverse=True)
        best_score, best_slot = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else 0.0
        confidence = max(0.0, min(1.0, (best_score - second_score) / max(best_score, 1.0)))
        if best_slot == 0 or confidence < 0.08:
            return 0.0, round(confidence, 4)
        _, effective = self._pickup_slots(
            best_slot * beats_per_measure / slots_per_measure,
            beats_per_measure,
            slots_per_measure,
        )
        return effective, round(confidence, 4)

    def _build_musicxml(
        self,
        events: dict[str, list[float]],
        tempo: float,
        grid: str,
        path: Path,
        duration_seconds: Optional[float] = None,
        beats_per_measure: int = 4,
        beat_unit: int = 4,
        measures_per_system: int = 3,
        pickup_beats: float = 0.0,
        notation_offset_seconds: float = 0.0,
        beat_times: Optional[list[float]] = None,
        timing_mode: str = "fixed",
    ) -> None:
        from music21 import clef, duration as m21dur, layout, meter, note, percussion, stream, tempo as m21tempo

        frac = GRID_Q[grid]
        step_seconds = float(frac) * 60.0 / float(tempo)
        measure_quarter_length = Fraction(beats_per_measure * 4, beat_unit)
        slots_per_measure_fraction = measure_quarter_length / frac
        if slots_per_measure_fraction.denominator != 1:
            raise ValueError("grid must divide the selected time signature into whole slots")
        slots_per_measure = int(slots_per_measure_fraction)
        slots_per_beat = max(1, round(slots_per_measure / beats_per_measure))
        mapped_beats = [float(value) for value in (beat_times or [])]
        use_beat_map = timing_mode == "beat_map" and len(mapped_beats) >= 2
        by_offset: dict[int, set[str]] = {}
        for name, values in events.items():
            for value in values:
                value = float(value)
                if use_beat_map:
                    beat_index = bisect.bisect_right(mapped_beats, value) - 1
                    if beat_index < 0:
                        continue
                    beat_index = min(beat_index, len(mapped_beats) - 2)
                    interval = max(1e-6, mapped_beats[beat_index + 1] - mapped_beats[beat_index])
                    subdivision = round((value - mapped_beats[beat_index]) / interval * slots_per_beat)
                    shifted_slot = beat_index * slots_per_beat + subdivision
                else:
                    shifted_slot = round((value - float(notation_offset_seconds)) / step_seconds)
                if shifted_slot >= 0:
                    by_offset.setdefault(shifted_slot, set()).add(name)

        def unpitched(name: str):
            step, octave, head = STAFF_MAP[name]
            value = note.Unpitched()
            value.displayStep = step
            value.displayOctave = octave
            if head != "normal":
                value.notehead = head
            return value

        pickup_slots, _ = self._pickup_slots(pickup_beats, beats_per_measure, slots_per_measure)
        if use_beat_map:
            source_slots = max(1, (len(mapped_beats) - 1) * slots_per_beat)
        else:
            notation_duration = max(0.0, float(duration_seconds or 0) - float(notation_offset_seconds))
            source_slots = int(math.ceil(notation_duration / step_seconds))
        event_slots = max(by_offset, default=-1) + 1
        total_slots = max(1, source_slots, event_slots)
        remaining_slots = max(0, total_slots - pickup_slots)
        full_measure_count = max(1, int(math.ceil(remaining_slots / slots_per_measure)))

        # Build complete measures explicitly. Relying on music21 to infer measures
        # from sparse absolute offsets can create an irregular pickup-like first bar
        # and makes cursor timing depend on the final detected hit.
        part = stream.Part()
        part.partName = "Drums"
        measure_specs: list[tuple[int, int, int, bool]] = []
        if pickup_slots:
            measure_specs.append((0, 0, pickup_slots, True))
        for full_index in range(full_measure_count):
            measure_specs.append((full_index + 1, pickup_slots + full_index * slots_per_measure, slots_per_measure, False))

        for sequence_index, (measure_number, start_slot, slot_count, is_pickup) in enumerate(measure_specs):
            measure = stream.Measure(number=measure_number)
            if sequence_index == 0:
                measure.insert(0, clef.PercussionClef())
                measure.insert(0, meter.TimeSignature(f"{beats_per_measure}/{beat_unit}"))
                measure.insert(0, m21tempo.MetronomeMark(number=round(tempo, 2)))
            full_index = measure_number - 1
            if not is_pickup and full_index > 0 and full_index % measures_per_system == 0:
                measure.insert(0, layout.SystemLayout(isNew=True))

            active_slots = [
                local_slot for local_slot in range(slot_count)
                if by_offset.get(start_slot + local_slot)
            ]
            if not active_slots:
                whole_measure_rest = note.Rest()
                whole_measure_rest.duration = m21dur.Duration(Fraction(slot_count) * frac)
                measure.insert(0, whole_measure_rest)
            else:
                if active_slots[0] > 0:
                    opening_rest = note.Rest()
                    opening_rest.duration = m21dur.Duration(Fraction(active_slots[0]) * frac)
                    measure.insert(0, opening_rest)
                for index, local_slot in enumerate(active_slots):
                    global_slot = start_slot + local_slot
                    names = sorted(by_offset[global_slot])
                    next_slot = active_slots[index + 1] if index + 1 < len(active_slots) else slot_count
                    gap_slots = max(1, next_slot - local_slot)
                    duration_slots = min(gap_slots, slots_per_beat)
                    if len(names) == 1:
                        element = unpitched(names[0])
                    else:
                        element = percussion.PercussionChord([unpitched(name) for name in names])
                    # The grid controls onset snapping, not a fixed printed duration.
                    # Extend each rhythmic event to the next onset so eighth/sixteenth
                    # patterns can beam together instead of being separated by tiny rests.
                    element.duration = m21dur.Duration(Fraction(duration_slots) * frac)
                    measure.insert(Fraction(local_slot) * frac, element)
                    remaining_slots = gap_slots - duration_slots
                    if remaining_slots > 0:
                        consolidated_rest = note.Rest()
                        consolidated_rest.duration = m21dur.Duration(Fraction(remaining_slots) * frac)
                        measure.insert(Fraction(local_slot + duration_slots) * frac, consolidated_rest)
            if is_pickup:
                measure.padAsAnacrusis()
                measure.showNumber = stream.enums.ShowNumber.NEVER
            part.append(measure)
        part.makeBeams(inPlace=True, failOnNoTimeSignature=False)
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
        if name == "musicxml" and int(task.get("score_version") or 0) < SCORE_VERSION:
            self._upgrade_musicxml(task_id)
        path = self.root / task_id / allowed[name]
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _upgrade_musicxml(self, task_id: str) -> None:
        """Lazily regenerate older scores without GPU work."""
        with self.lock:
            task = self.tasks[task_id]
            if int(task.get("score_version") or 0) >= SCORE_VERSION:
                return
            folder = self.root / task_id
            payload = json.loads((folder / "events.json").read_text(encoding="utf-8"))
            options = task.setdefault("options", {})
            pickup_mode = payload.get("pickup_mode", options.get("pickup_mode", "auto"))
            stored_notation_tempo = payload.get("notation_tempo") or options.get("notation_tempo")
            stored_mode = payload.get("notation_tempo_mode") or options.get("notation_tempo_mode")
            if stored_notation_tempo is None or (
                stored_mode != "manual"
                and abs(float(stored_notation_tempo) - float(payload["tempo"])) < 1e-6
            ):
                notation_tempo = self._refine_notation_tempo(
                    payload["events"], float(payload["tempo"]),
                    payload.get("grid", options.get("grid", "1/16")),
                )
                notation_tempo_mode = "auto"
            else:
                notation_tempo = float(stored_notation_tempo)
                notation_tempo_mode = stored_mode or "manual"
            notation_offset_seconds = float(payload.get(
                "notation_offset_seconds",
                options.get("notation_offset_seconds")
                if options.get("notation_offset_seconds") is not None
                else self._default_notation_offset(payload["events"]),
            ))
            raw_beat_times = self._detect_beat_times_for_existing_task(task_id, payload)
            timing_mode = payload.get("timing_mode", options.get("timing_mode", "beat_map"))
            if timing_mode == "beat_map" and len(raw_beat_times) < 2:
                timing_mode = "fixed"
            score_beat_times = self._align_beat_times(
                raw_beat_times, notation_offset_seconds,
                float(payload.get("duration") or task.get("duration") or 0),
            ) if timing_mode == "beat_map" else []
            pickup_beats, pickup_confidence = self._resolve_pickup(
                payload["events"], notation_tempo,
                payload.get("grid", options.get("grid", "1/16")),
                int(payload.get("beats_per_measure", options.get("beats_per_measure", 4))),
                int(payload.get("beat_unit", options.get("beat_unit", 4))),
                pickup_mode, float(payload.get("pickup_beats", options.get("pickup_beats", 0))),
                score_beat_times,
            )
            self._build_musicxml(
                payload["events"],
                notation_tempo,
                payload.get("grid", task.get("options", {}).get("grid", "1/16")),
                folder / "score.musicxml",
                duration_seconds=float(payload.get("duration") or task.get("duration") or 0),
                beats_per_measure=int(payload.get("beats_per_measure", task.get("options", {}).get("beats_per_measure", 4))),
                beat_unit=int(payload.get("beat_unit", task.get("options", {}).get("beat_unit", 4))),
                measures_per_system=int(payload.get("measures_per_system", task.get("options", {}).get("measures_per_system", 3))),
                pickup_beats=pickup_beats,
                notation_offset_seconds=notation_offset_seconds,
                beat_times=score_beat_times,
                timing_mode=timing_mode,
            )
            options.update({
                "pickup_mode": pickup_mode,
                "pickup_beats": pickup_beats,
                "pickup_confidence": pickup_confidence,
                "measures_per_system": int(payload.get("measures_per_system", options.get("measures_per_system", 3))),
                "notation_offset_seconds": notation_offset_seconds,
                "notation_tempo": notation_tempo,
                "notation_tempo_mode": notation_tempo_mode,
                "timing_mode": timing_mode,
            })
            payload.update({
                "pickup_mode": pickup_mode,
                "pickup_beats": pickup_beats,
                "pickup_confidence": pickup_confidence,
                "measures_per_system": options["measures_per_system"],
                "notation_offset_seconds": notation_offset_seconds,
                "notation_tempo": notation_tempo,
                "notation_tempo_mode": notation_tempo_mode,
                "timing_mode": timing_mode,
                "beat_times": raw_beat_times,
                "score_beat_times": score_beat_times,
            })
            (folder / "events.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            task["score_version"] = SCORE_VERSION
            task["updated_at"] = _now()
            self._write(task)

    def update_notation(self, task_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Persist notation-only changes and rebuild MusicXML without GPU inference."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.get("status") != "completed":
                raise ValueError("Task must be completed before notation can be edited")
            options = task["options"]
            grid = params.get("grid", options.get("grid", "1/16"))
            beats_per_measure = int(params.get("beats_per_measure", options.get("beats_per_measure", 4)))
            beat_unit = int(params.get("beat_unit", options.get("beat_unit", 4)))
            pickup_mode = params.get("pickup_mode", options.get("pickup_mode", "auto"))
            pickup_beats_input = float(params.get("pickup_beats", options.get("pickup_beats", 0)))
            measures_per_system = int(params.get("measures_per_system", options.get("measures_per_system", 3)))
            notation_offset_seconds = float(params.get(
                "notation_offset_seconds", options.get("notation_offset_seconds", 0)
            ))
            notation_tempo = float(params.get(
                "notation_tempo", options.get("notation_tempo") or task.get("tempo")
            ))
            timing_mode = params.get("timing_mode", options.get("timing_mode", "beat_map"))
            if pickup_mode not in ("none", "manual", "auto"):
                raise ValueError("pickup_mode must be none, manual, or auto")
            if grid not in GRID_Q:
                raise ValueError(f"grid must be one of {list(GRID_Q)}")
            if not 1 <= beats_per_measure <= 32:
                raise ValueError("beats_per_measure must be between 1 and 32")
            if beat_unit not in (1, 2, 4, 8, 16, 32):
                raise ValueError("beat_unit must be one of 1, 2, 4, 8, 16, or 32")
            if (Fraction(beats_per_measure * 4, beat_unit) / GRID_Q[grid]).denominator != 1:
                raise ValueError("grid must divide the selected time signature into whole slots")
            if not 1 <= measures_per_system <= 6:
                raise ValueError("measures_per_system must be between 1 and 6")
            if pickup_beats_input < 0 or pickup_beats_input >= beats_per_measure:
                raise ValueError("pickup_beats must be at least 0 and less than beats_per_measure")
            if notation_offset_seconds < 0 or notation_offset_seconds >= float(task.get("duration") or float("inf")):
                raise ValueError("notation_offset_seconds must be within the audio duration")
            if not 20 <= notation_tempo <= 999:
                raise ValueError("notation_tempo must be between 20 and 999 BPM")
            if timing_mode not in ("beat_map", "fixed"):
                raise ValueError("timing_mode must be beat_map or fixed")

            folder = self.root / task_id
            payload = json.loads((folder / "events.json").read_text(encoding="utf-8"))
            raw_beat_times = self._detect_beat_times_for_existing_task(task_id, payload)
            if timing_mode == "beat_map" and len(raw_beat_times) < 2:
                raise ValueError("No beat map is available; use fixed timing mode")
            score_beat_times = self._align_beat_times(
                raw_beat_times, notation_offset_seconds,
                float(payload.get("duration") or task.get("duration") or 0),
            ) if timing_mode == "beat_map" else []
            pickup_beats, confidence = self._resolve_pickup(
                payload["events"], notation_tempo, grid,
                beats_per_measure, beat_unit, pickup_mode, pickup_beats_input,
                score_beat_times,
            )
            temp_score = folder / "score.tmp.musicxml"
            self._build_musicxml(
                payload["events"], notation_tempo, grid, temp_score,
                duration_seconds=float(payload.get("duration") or task.get("duration") or 0),
                beats_per_measure=beats_per_measure, beat_unit=beat_unit,
                measures_per_system=measures_per_system, pickup_beats=pickup_beats,
                notation_offset_seconds=notation_offset_seconds,
                beat_times=score_beat_times,
                timing_mode=timing_mode,
            )
            os.replace(temp_score, folder / "score.musicxml")
            options.update({
                "pickup_mode": pickup_mode, "pickup_beats": pickup_beats,
                "pickup_confidence": confidence, "measures_per_system": measures_per_system,
                "notation_offset_seconds": notation_offset_seconds,
                "grid": grid, "beats_per_measure": beats_per_measure, "beat_unit": beat_unit,
                "notation_tempo": notation_tempo,
                "notation_tempo_mode": "manual",
                "timing_mode": timing_mode,
            })
            payload.update({
                "pickup_mode": pickup_mode, "pickup_beats": pickup_beats,
                "pickup_confidence": confidence, "measures_per_system": measures_per_system,
                "notation_offset_seconds": notation_offset_seconds,
                "grid": grid, "beats_per_measure": beats_per_measure, "beat_unit": beat_unit,
                "notation_tempo": notation_tempo,
                "notation_tempo_mode": "manual",
                "timing_mode": timing_mode,
                "beat_times": raw_beat_times,
                "score_beat_times": score_beat_times,
            })
            (folder / "events.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            task["score_version"] = SCORE_VERSION
            task["updated_at"] = _now()
            self._write(task)
            return self._public(task)

    def update_events(self, task_id: str, events: dict[str, list[float]]) -> dict[str, Any]:
        """Persist main-workspace MIDI edits back into a durable task."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.get("status") != "completed":
                raise ValueError("Task must be completed before events can be edited")
            normalized = {
                name: sorted(round(float(value), 5) for value in events.get(name, []) if float(value) >= 0)
                for name in CH_NAMES
            }
            folder = self.root / task_id
            payload = json.loads((folder / "events.json").read_text(encoding="utf-8"))
            payload["events"] = normalized
            payload["counts"] = {name: len(values) for name, values in normalized.items()}
            (folder / "events.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self._build_midi(normalized, float(payload["tempo"]), folder / "performance.mid")
            task["hit_counts"] = payload["counts"]
            task["score_version"] = 0  # lazily rebuild MusicXML on the next task/score read
            task["updated_at"] = _now()
            self._write(task)
            return self._public(task)

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
        paper_size: str = "fit",
        fps: int = 30,
    ) -> dict[str, Any]:
        self._validate_port(service_port)
        task = self.get(task_id)
        if task["status"] != "completed":
            raise ValueError("The score task must be completed before recording")
        if paper_size not in ("fit", "small", "medium", "large"):
            raise ValueError("paper_size must be fit, small, medium, or large")
        fps = max(12, min(60, int(fps)))
        self._validate_recording_browser()
        self._validate_offline_renderer()
        width, height = self._video_dimensions(aspect_ratio, width, height)
        recording_id = uuid.uuid4().hex
        recording = {
            "id": recording_id,
            "status": "queued",
            "width": width,
            "height": height,
            "paper_format": "A4_P",
            "paper_size": paper_size,
            "fps": fps,
            "container": "mp4",
            "video_codec": "h264_nvenc",
            "progress": 0.0,
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

    def _recording_launch_kwargs(self, playwright) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "headless": True,
            "args": ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"],
        }
        if self.recording_browser_executable:
            kwargs["executable_path"] = self.recording_browser_executable
        return kwargs

    def _validate_recording_browser(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ValueError(
                f"Playwright is not installed. Run: {self.python} -m pip install playwright"
            ) from None
        try:
            with sync_playwright() as playwright:
                executable = self.recording_browser_executable or playwright.chromium.executable_path
                if not executable or not Path(executable).is_file():
                    raise ValueError(
                        "Recording Chromium is not installed for the DrumLab service user. "
                        f"Run as that same user: {self.python} -m playwright install chromium. "
                        "Alternatively start DrumLab with --recording-browser /path/to/chromium."
                    )
                browser = playwright.chromium.launch(**self._recording_launch_kwargs(playwright))
                browser.close()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "Recording browser could not start. Install Chromium and its Linux dependencies "
                f"with: {self.python} -m playwright install --with-deps chromium. Details: {exc}"
            ) from None

    @staticmethod
    def _validate_offline_renderer() -> None:
        if importlib.util.find_spec("PIL") is None:
            raise ValueError("Offline recording requires Pillow. Run: pip install Pillow")
        if not shutil.which("ffmpeg"):
            raise ValueError("Offline recording requires FFmpeg on PATH")
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=False,
        )
        if encoders.returncode or "h264_nvenc" not in encoders.stdout:
            raise ValueError(
                "MP4 recording requires an FFmpeg build with NVIDIA NVENC "
                "(h264_nvenc) support"
            )

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
        self._record_offline(task_id, recording_id)

    def _record_offline(self, task_id: str, recording_id: str) -> None:
        from PIL import Image
        from playwright.sync_api import sync_playwright

        recording = self.get_recording(task_id, recording_id)
        log_prefix = f"[recording {task_id[:8]}/{recording_id[:8]}]"
        def record_log(message: str) -> None:
            print(f"{log_prefix} {message}", flush=True)

        folder = self.root / task_id / "recordings" / recording_id
        folder.mkdir(parents=True, exist_ok=True)
        self._record_update(task_id, recording_id, status="running")
        record_log(
            f"start {recording['width']}x{recording['height']} "
            f"{recording.get('fps', 30)} FPS MP4/NVENC"
        )
        score_png = folder / "score.png"
        page_pngs: list[Path] = []
        tile_tops: list[int] = []
        engrave_started = time.monotonic()
        record_log("engraving score in Chromium")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(**self._recording_launch_kwargs(playwright))
            context = browser.new_context(
                viewport={"width": recording["width"], "height": recording["height"]},
                # Engrave at 2x physical resolution, then downsample with Lanczos.
                # This keeps staff lines and note heads crisp after VP9 encoding.
                device_scale_factor=2,
            )
            page = context.new_page()
            page.goto(
                f"http://127.0.0.1:{self.port}/tasks/{task_id}"
                f"?recording=offline&paper_size={recording.get('paper_size', 'fit')}",
                wait_until="networkidle",
            )
            page.wait_for_function("window.__DRUMLAB_SCORE_READY__ === true", timeout=120000)
            timeline = page.evaluate("window.__DRUMLAB_EXPORT_TIMELINE__()")
            capture_height = int(timeline.get("captureHeight") or timeline.get("height") or 0)
            tile_height = int(recording["height"])
            if capture_height <= 0:
                raise RuntimeError("OSMD did not expose a valid capture height")
            max_scroll = max(0, capture_height - tile_height)
            tile_tops = list(range(0, max_scroll + 1, tile_height))
            if not tile_tops or tile_tops[-1] != max_scroll:
                tile_tops.append(max_scroll)
            for tile_top in tile_tops:
                actual_top = page.evaluate(
                    """top => {
                        const viewport = document.getElementById('viewport');
                        viewport.style.scrollBehavior = 'auto';
                        viewport.scrollTo(0, top);
                        return viewport.scrollTop;
                    }""",
                    tile_top,
                )
                if abs(float(actual_top) - tile_top) > 1.0:
                    raise RuntimeError(
                        f"Score tile scroll mismatch: requested {tile_top}, got {actual_top}"
                    )
                page.evaluate(
                    "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
                )
                page_png = folder / f"score-tile-{tile_top}.png"
                page.locator("#viewport").screenshot(path=str(page_png))
                page_pngs.append(page_png)
            context.close()
            browser.close()
        record_log(
            f"{len(page_pngs)} score tile(s) captured in "
            f"{time.monotonic() - engrave_started:.1f}s"
        )

        points = timeline.get("points") or []
        if not points:
            raise RuntimeError("The score did not expose any cursor positions")
        css_score_size = (int(recording["width"]), int(timeline["captureHeight"]))
        score = Image.new("RGB", css_score_size, (255, 255, 255))
        resampling = getattr(Image, "Resampling", Image)
        tile_height = int(recording["height"])
        for tile_top, page_png in zip(tile_tops, page_pngs):
            tile = Image.open(page_png).convert("RGB")
            expected_size = (int(recording["width"]), tile_height)
            if tile.size != expected_size:
                tile = tile.resize(expected_size, resampling.LANCZOS)
            remaining = css_score_size[1] - tile_top
            if remaining < tile.height:
                tile = tile.crop((0, 0, tile.width, max(0, remaining)))
            if tile.height > 0:
                score.paste(tile, (0, tile_top))
        width, height = int(recording["width"]), int(recording["height"])
        fps = int(recording.get("fps", 30))
        audio_duration = max(
            0.1,
            float(timeline.get("duration") or self.tasks[task_id].get("duration") or 0),
        )
        duration = audio_duration
        final = folder / "dynamic-score.mp4"
        score.save(score_png)
        cursor_png = folder / "cursor.png"
        filter_script = folder / "recording-filter.txt"
        cursor_width = max(4, round(float(np.median([p.get("width", 4) for p in points]))))
        cursor_height = max(12, round(float(np.median([p.get("height", 20) for p in points]))))
        Image.new("RGBA", (cursor_width, cursor_height), (66, 214, 111, 92)).save(cursor_png)

        # OSMD can vary a cursor box's y/height within one drum staff when a
        # chord spans high and low instruments. Cluster all cursor centers into
        # staff systems, then lock every point in a system to one stable y.
        point_centers = sorted(
            float(point["y"]) + float(point.get("height", cursor_height)) / 2
            for point in points
        )
        system_clusters: list[list[float]] = []
        cluster_gap = max(40.0, cursor_height * 2.0)
        for center in point_centers:
            if not system_clusters or center - system_clusters[-1][-1] > cluster_gap:
                system_clusters.append([center])
            else:
                system_clusters[-1].append(center)
        system_centers = [float(np.median(cluster)) for cluster in system_clusters]

        def stable_cursor_y(point: dict[str, Any]) -> float:
            center = float(point["y"]) + float(point.get("height", cursor_height)) / 2
            system_center = min(system_centers, key=lambda value: abs(value - center))
            return system_center - cursor_height / 2

        # A fit-to-width sheet can be a few CSS pixels wider than the even video
        # dimensions. Pad only when the sheet is narrower; otherwise crop the
        # wider sheet symmetrically. Use the same offset for cursor coordinates.
        sheet_x = float(timeline.get("wrapX") or 0)
        sheet_origin_y = float(timeline.get("wrapY") or 0)
        sheet_top = 12
        max_scroll = max(0, score.height - height + sheet_top * 2)
        samples: list[tuple[float, float, float, float]] = []
        for point in points:
            timestamp = max(0.0, min(duration, float(point["t"])))
            target_scroll = max(0.0, min(max_scroll, float(point["y"]) - height * 0.30))
            samples.append((
                timestamp,
                target_scroll,
                sheet_x + float(point["x"]),
                sheet_origin_y + stable_cursor_y(point),
            ))
        samples.sort(key=lambda item: item[0])
        collapsed_samples: list[tuple[float, float, float, float]] = []
        for sample in samples:
            if collapsed_samples and abs(sample[0] - collapsed_samples[-1][0]) <= 0.0001:
                collapsed_samples[-1] = sample
            else:
                collapsed_samples.append(sample)
        samples = collapsed_samples

        def step_expression(value_index: int) -> str:
            expression = f"{samples[0][value_index]:.3f}"
            previous = samples[0][value_index]
            previous_time = samples[0][0]
            for sample in samples[1:]:
                timestamp, value = sample[0], sample[value_index]
                if timestamp <= previous_time + 0.0001:
                    previous = value
                    continue
                delta = value - previous
                if abs(delta) >= 0.01:
                    expression += f"+({delta:.3f})*gte(t,{timestamp:.4f})"
                previous, previous_time = value, timestamp
            return expression

        scroll_expression = f"{samples[0][1]:.3f}"
        previous_scroll = samples[0][1]
        previous_time = samples[0][0]
        for timestamp, target_scroll, _, _ in samples[1:]:
            if timestamp <= previous_time + 0.0001:
                previous_scroll = target_scroll
                continue
            delta = target_scroll - previous_scroll
            if abs(delta) >= 0.5:
                scroll_expression += (
                    f"+({delta:.3f})*clip((t-{timestamp:.4f})/0.8,0,1)"
                )
            previous_scroll, previous_time = target_scroll, timestamp
        cursor_x_expression = step_expression(2)
        cursor_page_y_expression = step_expression(3)
        cursor_y_expression = f"({cursor_page_y_expression})-({scroll_expression})"
        padded_width = max(width, score.width)
        padded_height = max(height, score.height)
        filter_script.write_text(
            f"[0:v]pad={padded_width}:{padded_height}:(ow-iw)/2:0:white,"
            f"crop@score={width}:{height}:(iw-{width})/2:'{scroll_expression}'[background];\n"
            f"[background][1:v]overlay@cursor=x='{cursor_x_expression}':"
            f"y='{cursor_y_expression}':eval=frame:shortest=1,format=yuv420p[out]",
            encoding="utf-8",
        )
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-framerate", str(fps), "-i", str(score_png),
            "-loop", "1", "-framerate", str(fps), "-i", str(cursor_png),
            "-filter_complex_script", str(filter_script), "-map", "[out]",
            "-t", f"{duration:.6f}", "-r", str(fps), "-an", "-c:v", "h264_nvenc",
            "-preset", "p4", "-tune", "hq", "-rc", "vbr",
            "-cq", "20", "-b:v", "0", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(final),
        ]
        record_log("FFmpeg filter pipeline + NVENC started (no Python frame pipe)")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        last_progress_log = time.monotonic()
        rendered_seconds = 0.0
        assert process.stdout is not None
        for line in process.stdout:
            key, separator, value = line.strip().partition("=")
            if not separator:
                continue
            if key in ("out_time_us", "out_time_ms"):
                # Modern FFmpeg reports both fields in microseconds despite the
                # historical out_time_ms name.
                try:
                    progress_time = float(value)
                except (TypeError, ValueError):
                    progress_time = None
                if progress_time is not None and math.isfinite(progress_time):
                    rendered_seconds = max(rendered_seconds, progress_time / 1_000_000.0)
            now = time.monotonic()
            if now - last_progress_log >= 5.0:
                progress = min(1.0, rendered_seconds / duration)
                elapsed = max(0.001, now - engrave_started)
                effective_fps = rendered_seconds * fps / elapsed
                eta = max(0.0, duration - rendered_seconds) / max(
                    0.001, rendered_seconds / elapsed,
                )
                record_log(
                    f"FFmpeg {progress * 100:.1f}% "
                    f"({effective_fps:.1f} FPS, ETA {eta:.0f}s)"
                )
                self._record_update(
                    task_id, recording_id, progress=round(progress, 4),
                    rendered_seconds=round(rendered_seconds, 2),
                    render_fps=round(effective_fps, 2), eta_seconds=round(eta, 1),
                )
                last_progress_log = now
        stderr = process.stderr.read() if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg GPU recording failed: {stderr.strip()}")
        score_png.unlink(missing_ok=True)
        for page_png in page_pngs:
            page_png.unlink(missing_ok=True)
        cursor_png.unlink(missing_ok=True)
        filter_script.unlink(missing_ok=True)
        self._record_update(
            task_id, recording_id, status="completed", path=str(final), progress=1.0,
            rendered_seconds=round(duration, 2),
        )
        record_log(f"completed in {time.monotonic() - engrave_started:.1f}s: {final}")

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
