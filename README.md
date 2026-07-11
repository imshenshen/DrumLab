# DrumLab

![DrumLab](screenshot.png)

A GUI wrapper around [Demucs](https://github.com/adefossez/demucs) and
[ADTOF](https://github.com/xavriley/ADTOF-pytorch) that separates a song into
its individual parts — drums, bass, vocals, the rest — transcribes the drums
to MIDI and sheet music, and lets you adjust the speed so you can play along.

Runs as a local web server — everything happens on your own machine, fully
offline, nothing leaves your computer.

---

## Features

- **Play from your library** — point DrumLab at your music folders to build a searchable,
  artist-grouped song library, then select tracks straight from the browser.
- **Play queue** — pick from the library, drag files in directly or use **party shuffle**
  for play-along convenience.
- **Separate the track** — runs Demucs (full `htdemucs` separation) to split the song into
  drums / bass / other / vocals. Pick which sources to isolate; the rest can be summed into a
  single backing track.
- **Extract drum MIDI** — runs ADTOF over the audio to detect kick, snare, toms, hi-hat and
  cymbals, and turns them into MIDI. Per-instrument detection thresholds can be nudged
  on-the-fly — hits are re-picked instantly without re-running the model.
- **Verify & adjust** — listen back and decide for yourself whether you like the result.
  Make basic MIDI edits, loop a section, and dial in per-lane gain knobs, stereo level
  meters (dBFS), and a pitch-preserved playback-speed knob.
- **Export** — stems in FLAC / WAV / AAC / ALAC / AIFF / OGG, plus MIDI, quantized MIDI,
  and MusicXML. Optionally export at the current playback speed.
- **Persistent tasks + dynamic score** — queue server-local audio paths, receive a task ID,
  and open `/tasks/{task_id}` for audio-synchronised notation with a moving cursor and
  automatic scrolling.
- **Agent/MCP API** — Streamable HTTP MCP tools at `/mcp/` create and inspect tasks and
  request silent dynamic-score recordings.

---

## Requirements

Everything installs into a single Python environment:

- **Python 3.10+** — a dedicated `venv` or conda environment is strongly recommended.
- **[FFmpeg](https://ffmpeg.org/)** on your `PATH` — audio decode, format conversion, and
  time-stretching.
- **[PyTorch](https://pytorch.org/)** (`torch` + `torchaudio`) — install a CUDA build for
  GPU acceleration (see [Install](#install)). A CPU-only build works too, just much slower.
- **[Demucs](https://github.com/adefossez/demucs)** — source separation, invoked as
  `python -m demucs`.
- **[ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch)** — the drum-transcription
  model (imported as `adtof_pytorch`). DrumLab uses it as an installed package, or finds a
  sibling checkout at `../ADTOF-pytorch/src/adtof_pytorch`.
- **FastAPI + Uvicorn + python-multipart + music21** — the web server and MusicXML export.
  These are pure-Python and don't touch torch.
- **MCP Python SDK** — agent-facing Streamable HTTP tools.
- **Playwright Chromium** — only needed by the optional silent score-recording API.

---

## Install

> The one thing to get right is **PyTorch first**, so GPU support sticks. Both Demucs and
> ADTOF list `torch` as a dependency; if you let pip install them before torch (or run
> `pip install -U` afterwards), pip can quietly swap your CUDA build for a CPU-only one.
> Everything still runs — just on the CPU. So: install torch, then everything else, then
> verify CUDA is still there.

1. **Get the code** (DrumLab and, optionally, ADTOF-pytorch side by side):

   ```sh
   git clone https://github.com/DomekRomek/Drumlab.git
   git clone https://github.com/xavriley/ADTOF-pytorch.git   # optional — pip-install alternative below
   cd Drumlab
   ```

2. **Create and activate an environment:**

   ```sh
   python -m venv .venv
   .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
   ```

3. **Install PyTorch with CUDA.** Grab the exact command for your OS + CUDA version from
   the [official selector](https://pytorch.org/get-started/locally/). For example:

   ```sh
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
   ```

   (Drop the `--index-url` for a CPU-only build.)

4. **Install the rest**

    If you cloned ADTOF-pytorch as a sibling folder instead, you can skip the second command.
   ```sh
   pip install demucs fastapi uvicorn python-multipart music21 "mcp>=1.27,<2" playwright
   pip install --no-deps git+https://github.com/xavriley/ADTOF-pytorch.git
   playwright install chromium     # only required for automatic video recording
   ```

   <sub>*`--no-deps` stops pip from reinstalling torch over the CUDA build from step 3.*</sub>

5. **Verify CUDA (optional)**

   ```sh
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   If this prints `True`, you're set for GPU. If it flipped to `False`, you may want to try
   re-doing step 3.

6. **Install FFmpeg** if you don't have it ([download](https://ffmpeg.org/download.html)),
   and make sure `ffmpeg` is on your `PATH`.

---

## Run

```sh
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
python app.py
```

This starts the server on `127.0.0.1:8765` and opens it in your default browser.

| Flag | Effect |
| --- | --- |
| `--port N` | listen on a different port (default `8765`) |
| `--host ADDR` | bind address (default `127.0.0.1`; `0.0.0.0` to expose on your LAN) |
| `--no-browser` | don't open a browser window |
| `--preload` | download all Demucs models, then exit |
| `--library FOLDER` | index a folder for the song library / party shuffle (repeatable) |
| `--task-root FOLDER` | restrict agent-supplied audio paths to this folder (repeatable) |
| `--task-output-root FOLDER` | store persistent task folders and generated artifacts here |
| `--task-timeout SECONDS` | maximum whole-pipeline runtime per task (default 3600; 0 disables) |

> **Song library.** The party-shuffle / up-next browser indexes folders you point it at.
> Pass `--library FOLDER` (repeatable) to index one or more folders at startup — e.g.
> `python app.py --library "D:\Music"` — or start without it and pick a folder from the UI.
> With no library configured the browser just starts empty.

> **Single shared workspace.** DrumLab is a single-user tool: the server keeps one
> global state (current song, stems, transcription). Every browser that connects —
> including a second tab or another device when you bind with `--host 0.0.0.0` —
> sees and edits that *same* workspace. Loading a song on your phone replaces what's
> on your PC. This is by design; don't run two clients against one server expecting
> independent sessions.

## Agent tasks and dynamic scores

The task API is separate from the legacy shared GUI workspace. Tasks are persistent under
`workdir/tasks/<task-id>` by default, or under `<task-output-root>/<task-id>` when
`--task-output-root` is supplied. The GPU pipeline is intentionally single-file so concurrent
agents cannot start competing Demucs/ADTOF processes.

Example with separate input and output locations:

```sh
python app.py --host 0.0.0.0 --port 8765 --no-browser \
  --task-root /home/shenshen/runclave-inputs \
  --task-output-root /home/shenshen/runclave-outputs \
  --task-timeout 1800
```

`--task-root` controls which source audio files agents may read. `--task-output-root`
controls where DrumLab writes task metadata, logs, decoded audio, stems, activation caches,
MusicXML, MIDI, and silent recordings.

`--task-timeout` covers the complete processing lifetime after a task leaves the queue:
FFmpeg ingest, optional Demucs separation, ADTOF inference, and notation generation. On
timeout DrumLab terminates the active process group and marks the task `timed_out`. A value
of `0` disables automatic timeouts.

Create a task from an absolute path that exists **on the DrumLab server**:

```sh
curl -X POST http://127.0.0.1:8765/api/tasks \
  -H 'content-type: application/json' \
  -d '{"audio_path":"/music/song.flac","service_port":8765}'
```

If the source is already a drum-only recording, skip Demucs and start directly at ADTOF:

```sh
curl -X POST http://127.0.0.1:8765/api/tasks \
  -H 'content-type: application/json' \
  -d '{"audio_path":"/music/drums.wav","service_port":8765,"source_mode":"drum_only"}'
```

`source_mode` accepts `full_mix` (default, Demucs then ADTOF) or `drum_only` (decode then
ADTOF directly). The same argument is available in the MCP `create_drum_score_task` tool
and in the `/demo` form.

Set the notation time signature with `beats_per_measure` and `beat_unit`; both default to
`4`, producing 4/4. For example, `{"beats_per_measure":6,"beat_unit":8,"grid":"1/16"}`
produces a 6/8 score quantized to sixteenth-note positions. The time signature controls
measure structure, while `grid` independently controls the smallest timing snap. Tempo is
still expressed as quarter-note BPM, matching the ADTOF estimate.

`measures_per_system` controls printed score density and defaults to `3`. DrumLab writes
explicit MusicXML system breaks, so the same number of measures per line is used in the
dynamic page, A4 printing, and silent recordings.

The quantization grid snaps onset positions; it is not assigned as every note's printed
duration. DrumLab derives rhythmic duration from the next onset, consolidates silent spans,
and applies beat-aware beaming so continuous eighth/sixteenth patterns stay visually grouped.

`notation_offset_seconds` separates audio lead-in from musical notation. By default DrumLab
sets it to the first quantized detected drum onset: audio playback still begins at 0, while
that first onset is placed at score time 0 (the first beat unless a pickup is configured).
The dynamic page can adjust and save this offset; cursor timing subtracts the same value from
audio time so playback stays synchronized.

Pickup handling uses `pickup_mode`: `none`, `manual`, or `auto` (default). Manual mode uses
`pickup_beats`; auto mode scores kick, snare, and cymbal bar phases and stores both the
detected pickup length and confidence. On a completed task, the dynamic-score page can edit
the pickup and measures-per-system settings and save them back. REST clients can call
`PATCH /api/tasks/{task_id}/notation`; MCP clients use `update_drum_score_notation`. This
rebuilds only MusicXML and does not rerun Demucs or ADTOF.

The main workbench can open a completed task by ID from its Export panel, or directly with
`/?task_id=TASK_ID`. It loads the task audio and detected hits into the normal waveform/MIDI
editor without rerunning either model. In task mode, MusicXML, MIDI, Sheet Music, meter,
pickup, lead-in, grid, and bars-per-line all use the durable task. MIDI roll edits are written
back to `events.json` and MIDI immediately; MusicXML is rebuilt lazily from those edits.

The response contains a task ID and `queued` or `running` status. Poll
`GET /api/tasks/{task_id}` until `completed`, then open `/tasks/{task_id}`. Artifacts are
available from the URL map in the task response. The supplied `service_port` must match the
actual DrumLab port; this prevents an agent from accidentally submitting work to the wrong
local service. Use one or more `--task-root` flags to prevent agents from reading audio
outside approved directories.

For manual operation, open `http://HOST:PORT/demo`. This page submits local-path tasks,
polls the complete task queue, opens completed dynamic scores, exports MusicXML/MIDI, and
starts or downloads silent WebM recordings in 16:9, 9:16, 4:3, or 1:1.

MCP clients connect to `http://HOST:PORT/mcp/` and receive these tools:

- `create_drum_score_task`
- `get_drum_score_task`
- `stop_drum_score_task`
- `update_drum_score_notation`
- `create_dynamic_score_recording`
- `get_dynamic_score_recording`

REST clients stop a queued or running task with `POST /api/tasks/{task_id}/stop`. Queued
tasks are marked `cancelled` immediately. Running tasks terminate their FFmpeg, Demucs, or
ADTOF process group before being marked `cancelled`, releasing associated CUDA resources.

Create a silent recording after the task completes:

```sh
curl -X POST http://127.0.0.1:8765/api/tasks/TASK_ID/recordings \
  -H 'content-type: application/json' \
  -d '{"service_port":8765,"aspect_ratio":"9:16","width":1080,"paper_size":"fit"}'
```

Recording runs asynchronously and produces a WebM without audio. Poll the returned recording
URL until `completed`, then download its `url` field.

The video `aspect_ratio` and dimensions control only the WebM canvas. Score engraving always
uses portrait A4 (`A4_P`) inside that canvas. `paper_size` controls only the displayed A4
zoom and accepts `fit`, `small`, `medium`, or `large`; it does not change measure/system
breaks. The same A4 display-size choices are available on `/demo` and `/tasks/{task_id}`.

---

## Structure

```
app.py                 FastAPI server + pipeline orchestration + exports
adtof_worker.py        ADTOF inference, run as a separate killable subprocess
requirements-extra.txt DrumLab's own (non-torch) dependencies
static/                single-page UI (index.html, app.js, style.css, score viewer)
static/vendor/         pinned local copies of WaveSurfer, Tone.js, OSMD (no CDN)
workdir/               runtime caches + generated downloads — safe to delete
```

## About

This started from wanting a slick way to prep songs to play along to and slow them down
to my skill level — on my own machine, without having to fiddle with my DAW every time.
As an intermediate drummer I just wanted one tool that did exactly what I wanted,
but after ironing wrinkles enough times it turned into something pleasant for me
to use, so I figured I'd put it out there in case it's useful to someone else.

## Contributors

- **DomekRomek** — system architecture, testing, debugging, UI/UX.
- **Claude** — code writing.

## Credits

Built on [Demucs](https://github.com/adefossez/demucs) for source separation and
[ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) for drum transcription, with
[music21](https://web.mit.edu/music21/) for MusicXML, [FastAPI](https://fastapi.tiangolo.com/)
and [Uvicorn](https://www.uvicorn.org/) running the server, and
[WaveSurfer.js](https://wavesurfer.xyz/), [Tone.js](https://tonejs.github.io/), and
[OpenSheetMusicDisplay](https://opensheetmusicdisplay.org/) for the UI.

**Thank you to the developers of these projects for making my life easier.**
