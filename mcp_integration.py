# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP tools exposed by the DrumLab HTTP service."""

from __future__ import annotations

from typing import Any, Optional


def build_mcp_http_app(task_manager) -> tuple[Any, Any]:
    """Return (FastMCP instance, Streamable HTTP ASGI app), or (None, None)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        return None, None

    server = FastMCP(
        "DrumLab",
        instructions=(
            "Create automatic drum-transcription jobs from absolute audio paths on the "
            "DrumLab server, inspect job status, and request silent dynamic-score videos."
        ),
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
    )

    @server.tool()
    def create_drum_score_task(
        audio_path: str,
        service_port: int,
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
        fps: int = 100,
        thresholds: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        """Queue an audio-to-dynamic-drum-score task using a server-local absolute path."""
        return task_manager.submit(
            audio_path,
            service_port,
            model=model,
            device=device,
            source_mode=source_mode,
            grid=grid,
            beats_per_measure=beats_per_measure,
            beat_unit=beat_unit,
            measures_per_system=measures_per_system,
            pickup_mode=pickup_mode,
            pickup_beats=pickup_beats,
            notation_offset_seconds=notation_offset_seconds,
            fps=fps,
            thresholds=thresholds,
        )

    @server.tool()
    def get_drum_score_task(task_id: str) -> dict[str, Any]:
        """Return task state, timeout information, and artifact URLs."""
        return task_manager.get(task_id)

    @server.tool()
    def stop_drum_score_task(task_id: str) -> dict[str, Any]:
        """Stop a queued/running score task and terminate its child process group."""
        return task_manager.stop(task_id)

    @server.tool()
    def update_drum_score_notation(
        task_id: str,
        service_port: int,
        pickup_mode: str = "auto",
        pickup_beats: float = 0.0,
        measures_per_system: int = 3,
        notation_offset_seconds: Optional[float] = None,
        grid: str = "1/16",
        beats_per_measure: int = 4,
        beat_unit: int = 4,
    ) -> dict[str, Any]:
        """Re-detect or manually set pickup notation and rebuild MusicXML without GPU work."""
        task_manager._validate_port(service_port)
        return task_manager.update_notation(task_id, {
            "pickup_mode": pickup_mode,
            "pickup_beats": pickup_beats,
            "measures_per_system": measures_per_system,
            "grid": grid,
            "beats_per_measure": beats_per_measure,
            "beat_unit": beat_unit,
            **({"notation_offset_seconds": notation_offset_seconds} if notation_offset_seconds is not None else {}),
        })

    @server.tool()
    def create_dynamic_score_recording(
        task_id: str,
        service_port: int,
        aspect_ratio: str = "16:9",
        width: int = 1920,
        height: Optional[int] = None,
        paper_size: str = "fit",
    ) -> dict[str, Any]:
        """Queue a silent WebM recording of a completed dynamic score."""
        return task_manager.create_recording(
            task_id,
            service_port,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            paper_size=paper_size,
        )

    @server.tool()
    def get_dynamic_score_recording(task_id: str, recording_id: str) -> dict[str, Any]:
        """Return the status and download URL of a dynamic-score recording."""
        return task_manager.get_recording(task_id, recording_id)

    return server, server.streamable_http_app()
