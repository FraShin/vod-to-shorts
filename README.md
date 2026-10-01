# VOD → Shorts Pipeline

Turn a full stream VOD into vertical, subtitled clips — automatically.

Point it at a recording of a 4-hour stream and it produces a handful of
ready-to-post 1080×1920 clips: it listens to the audio to find the screams and
the hype peaks, reads the transcript to score what was *said*, asks an LLM to
pick the best moments, then cuts, stacks and burns subtitles on them.

Built for streamers who would rather stream than edit. Code comments are in
Italian; everything user-facing is in English.

---

## How it works

Eight stages, each one a separate process reading and writing JSON in `output/`.
Every stage can be re-run on its own, and finished stages are skipped, so a
failed run resumes instead of starting over.

| # | Stage | What it does | Artifact |
|---|-------|--------------|----------|
| 1 | `audio_engine.py` | Extracts the microphone track to 16 kHz mono WAV | `video.wav` |
| 2 | `transcribe_engine.py` | Whisper `large-v3` with word timestamps + RMS energy spikes | `transcription.json` |
| 3 | `merge_segments.py` | Glues fragmented phrases into speakable blocks | `merged_segments.json` |
| 4 | `audio_analysis.py` | **Cinema Engine**: RMS envelope, spikes, energy tails | `segments_with_audio.json` |
| 5 | `score_segments.py` | NLP scoring + dynamic top-30 % threshold | `scored_segments.json` |
| 6 | `qa_validator.py` | Sanitizes abuse of the spike cooldown | – |
| 7 | `judge_agent.py` | LLM picks the clips, with acoustic fallbacks | `final_choice.json` |
| 8 | `cut_engine.py` | Vertical render + ASS subtitle burn-in | `clips/*.mp4` |

Also included:

- **`manual_cutter.py`** — interactive clipping, same render path as stage 8.
- **`stable_ts_probe.py`** — an A/B experiment comparing stable-ts against
  faster-whisper word timings. Not part of the pipeline.
- **`tests/smoke_pipeline.py`** — runs in two seconds with no VOD, GPU or CUDA,
  and tells you whether the scoring chain is still intact.
- **`PIPELINE.md`** — the JSON contract between stages, field by field.

### What makes the selection good

Three independent signals feed the final choice:

1. **Acoustics** — an RMS envelope with adaptive thresholds (90th percentile
   for spikes, 25th for the silence floor, computed on *your* VOD). A shout
   surrounded by quiet becomes a `cinematic_spike` with a multiplier up to 4×.
2. **Text** — trigger categories with different weights, plus a bonus for
   questions and exclamation marks.
3. **An LLM judge** — sees the top 40 candidates and picks the most viral ones,
   writing a hook for each.

If the LLM picks too few, two fallbacks fire: **Spike Recovery** (acoustic peaks
the AI ignored) and **Hype Detector** (orphan segments containing your most
iconic phrases, capped to avoid flooding). No API key? Stages 1-6 still run and
the fallbacks alone produce clips.

---

## Requirements

- **Python 3.11+** (developed and tested on 3.12.3)
- **FFmpeg** with the `subtitles` filter (libass) — required for subtitle burn-in
- **NVIDIA GPU with CUDA** for Whisper on GPU and NVENC encoding
  (see [Running without an NVIDIA GPU](#running-without-an-nvidia-gpu))
- ~4 GB of disk for the `large-v3` model on first run

Tested on **Linux / WSL2** with CUDA 12.1. macOS and Windows are untested but
the code paths exist.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` explains how to get the **CUDA build of torch** — plain
`pip install torch` gives you the CPU-only wheel and makes transcription
painfully slow.

Verify everything at once:

```bash
python main.py --check
```

This checks Python, ffmpeg, ffprobe, NVENC, libass, torch, CUDA, your layout
maths, the VOD and the vocabulary, then exits without touching anything. Running
it first saves you from discovering a missing encoder twenty minutes in.

## Use

```bash
python main.py                        # latest .mp4 in input/, 10 clips
python main.py --vod stream.mp4       # a specific VOD
python main.py --clips 5              # how many clips
python main.py --no-wipe              # keep output/ and clips/ as they are
python main.py --check                # environment check
```

Run the smoke test:

```bash
python tests/smoke_pipeline.py
```

### Expected layout

```
your-project/
├── scripts/          ← this repository
├── input/            ← drop your VODs here
├── output/           ← intermediate JSON + video.wav
├── clips/            ← finished clips
├── clips_archive/    ← clips from previous runs, moved here each time
└── logs/             ← one log per stage
```

`input/`, `output/`, `clips/`, `clips_archive/` and `logs/` are created
automatically. **Note:** at startup `main.py` clears `output/` and moves the clips
from the previous run to `clips_archive/<date-time>/`, so a new run can never throw
away a clip you already produced — use `--no-wipe` to leave both untouched.

---

## Configure it for *your* stream

Two files, both optional, both gitignored. Neither requires touching the code.

### 1. `.env` — paths, credentials, technical knobs

```bash
cp .env.example .env
```

Everything is documented in `.env.example`, with defaults inline. The ones that
actually matter:

| Variable | Why you'd change it |
|---|---|
| `SOURCE_X/Y/W/H`, `TARGET_CAM_H/GAME_H` | **Almost certainly you must.** These crop your webcam and game out of the VOD. The defaults match one specific OBS layout — if yours differs, the clip will be cut wrong. Watch a frame of your VOD to find the numbers. |
| `AUDIO_MIC_STREAM_INDEX` / `AUDIO_GAME_STREAM_INDEX` | Which audio tracks are your mic and your game. Check with `ffprobe`. |
| `WHISPER_LANGUAGE` | Language spoken in the VOD. |
| `TARGET_TOTAL_CLIPS` | How many clips per run. |
| `USE_NVENC` | `0` to encode on CPU instead of NVIDIA hardware. |
| `FFMPEG_BIN` | Path to ffmpeg if it isn't on your `PATH` — Windows almost always needs this. |
| `DEEPSEEK_API_KEY` | Enables the LLM judge. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Optional "pipeline finished" notification. |

### 2. `vocabulary.json` — your taste

```bash
cp vocabulary.example.json vocabulary.json
```

This is the interesting part. Everything that depends on **who is streaming** —
your catchphrases, your triggers, your slang corrections, the context prompt
handed to Whisper, the persona given to the judging LLM — lives in this single
file instead of being hardcoded in six different scripts:

| Section | What it holds |
|---|---|
| `profile` | Your name, genres, style, extra selection rules for the LLM |
| `whisper_prompt` | Context so Whisper spells your in-jokes right instead of inventing similar words |
| `hype` | Your hype phrases — used to merge segments and to anchor cuts on the payoff |
| `triggers` | Words that score segments, in five weighted categories |
| `slang_fixes` | Regex corrections for what Whisper mishears |

Every field is explained inside the file itself. **Empty lists are fine** — the
pipeline still works using acoustics and punctuation alone; it just won't
recognise your catchphrases.

Because `vocabulary.json` is gitignored, a `git pull` will never overwrite your
values.

---

## Running on native Windows (no WSL)

The pipeline runs on Windows 11 natively too. Two things need attention.

**1. Install torch from the PyTorch index, not PyPI.** Same as on Linux, but
easier to get wrong here:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1` with *"running scripts is disabled
on this system"*, you don't have to touch your execution policy — skip
activation and call the venv's interpreter by path. This matters more than it
looks: a bare `pip` installs into your **global** Python when no venv is
active, which quietly leaves you with an empty virtualenv and packages
everywhere else.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

To fix activation properly, allow locally-created scripts for your user only
(no admin rights required):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

If `Get-ExecutionPolicy -List` shows a policy under `MachinePolicy` or
`UserPolicy`, that's set by Group Policy, it wins over the above, and the
by-path approach is the way to go.

**2. Tell it where ffmpeg is.** Windows machines almost never have it on
`PATH`. Download a build, then add to `.env`:

```
FFMPEG_BIN=C:/ffmpeg/bin/ffmpeg.exe
```

Forward slashes work and save you from escaping backslashes. `ffprobe` is then
looked up in the same folder automatically.

The CUDA runtime ships *inside* the pip wheels on Windows: `torch/lib` holds
cuBLAS and cuDNN, and `transcribe_engine.py` registers those directories with
`os.add_dll_directory` before CTranslate2 loads, so `faster-whisper` finds them
without you copying any DLL by hand.

Then verify and go:

```powershell
python main.py --check
python main.py --vod "C:\path\to\stream.mp4"
```

---

## Running without an NVIDIA GPU

Set `USE_NVENC=0` to encode with `libx264` on the CPU — slower, fully portable.

For transcription, `transcribe_engine.py` detects CUDA and falls back to CPU
automatically. It works, but `large-v3` on CPU is roughly an order of magnitude
slower; set `WHISPER_MODEL=small` or `medium` for a usable turnaround.

The CUDA libraries imported from pip are optional too: on a system-wide CUDA
install, on native Windows, or on macOS, the import is skipped instead of
crashing.

---

## Credits and licence

MIT — see [LICENSE](LICENSE). Issues and pull requests are welcome.

The interesting engineering here is in `audio_analysis.py` (adaptive RMS
envelope with energy tails) and in the three-tier selection cascade in
`judge_agent.py`. If you fork this for another genre, that is where the ideas
are.