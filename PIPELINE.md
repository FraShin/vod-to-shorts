# Pipeline data contract

Every stage is a separate process. They never call each other: they hand work
over through JSON files in `output/`. This document describes that handover —
which file each stage reads, which it writes, and which fields the next stage
actually depends on.

Knowing this contract is what lets you run a single stage, replace one, or
debug a run that went wrong at step 5 of 8.

```
VOD (.mp4)
   │
   ├─ audio_engine.py ──────► video.wav
   │                              │
   │                              ▼
   ├─ transcribe_engine.py ─► transcription.json ─┐
   │                                              │
   │                          merge_segments.py ◄─┘
   │                                  │
   │                                  ▼
   │                          merged_segments.json
   │                                  │
   │      audio_analysis.py ◄─────────┘  (legge anche video.wav)
   │              │
   │              ▼
   │      segments_with_audio.json
   │              │
   │              ▼
   │      score_segments.py ──► scored_segments.json
   │              │                      │
   │              ▼                      ▼
   │      qa_validator.py         judge_agent.py ──► final_choice.json
   │                                     │                  │
   └─────────────────────────────────────┴──────────────────┘
                                                          ▼
                                                  cut_engine.py
                                                          │
                                                          ▼
                                                  clips/*.mp4
```

All JSON is UTF-8, `ensure_ascii=False`. Every timestamp is **seconds as a
float** from the start of the source VOD.

---

## `video.wav` — written by `audio_engine.py`

Not JSON: a WAV file, 16 kHz mono, PCM signed 16-bit little-endian.

16 kHz mono is what Whisper wants, so no resampling happens later. Only the
microphone track is extracted (`AUDIO_EXTRACT_STREAM_INDEX`), because mixing in
game audio pollutes the transcript.

Skipped if the file already exists and is non-empty.

---

## `transcription.json` — written by `transcribe_engine.py`

```jsonc
{
  "source_vod": "stream.mp4",          // basename: usato per capire se la
  "source_vod_path": "/path/stream.mp4", // trascrizione appartiene al VOD scelto
  "segments": [
    {
      "start": 12.34,
      "end": 15.87,
      "text": "allora ragazzi buonasera [ENERGY_SPIKE]",
      "words": [
        { "start": 12.34, "end": 12.61, "word": "allora", "probability": 0.94 }
      ],
      "energy_spikes": [13.2, 14.75]
    }
  ]
}
```

| Field | Notes |
|---|---|
| `words[].start/end` | Clamped inside the parent segment — Whisper can emit word timings that spill outside it |
| `energy_spikes` | Timestamps of RMS peaks **inside this segment**. Metadata only: fake words are deliberately *not* injected into `words[]`, which would corrupt subtitle timings |
| `text` | Gets the energy marker appended when `energy_spikes` is non-empty |

The energy marker is an internal pipeline signal, **not** an utterance. It is
defined by `ENERGY_SPIKE_MARKER` in `config.py`. `ENERGY_SPIKE_TOKEN` is derived
from it and is the form to look for in normalised text (which strips brackets).

---

## `merged_segments.json` — written by `merge_segments.py`

A **top-level JSON array** (not an object).

```jsonc
[
  {
    "start": 12.34,
    "end": 18.02,
    "text": "allora ragazzi buonasera a nanna [ENERGY_SPIKE]",
    "words": [ /* same shape as above */ ],
    "energy_spikes": [13.2, 14.75]
  }
]
```

Whisper cuts on silence, which fragments sentences. This stage reglues them:

- normal speech: merges across gaps up to `BASE_MAX_GAP` (0.8 s)
- if either side contains a hype phrase or the energy marker: up to
  `HYPE_MAX_GAP` (4.0 s) — note this also pulls in the **setup before** the
  payoff, so a clip does not start mid-sentence
- a strong punctuation mark followed by a pause breaks the block
- `MAX_CLIP_DURATION` (42 s) caps the result

Hype phrases and words come from `vocabulary.json`. If it is absent those lists
are empty and merging happens on silence alone.

---

## `segments_with_audio.json` — written by `audio_analysis.py`

Same array shape, plus two fields per segment, and `end` may have been
**extended**:

```jsonc
[
  {
    "start": 12.34,
    "end": 20.15,              // esteso sulla coda energetica dell'urlo
    "text": "...",
    "words": [ /* l'ultima parola è riallineata alla nuova fine */ ],
    "energy_spikes": [13.2],
    "cinematic_spike": true,   // NUOVO
    "audio_multiplier": 3.41   // NUOVO
  }
]
```

| Field | Notes |
|---|---|
| `cinematic_spike` | `true` when a spike is preceded by quiet — a shout, not just loud game audio |
| `audio_multiplier` | `max_rms / spike_threshold`, capped at `MAX_MULTIPLIER` (4.0). Multiplies the final score in stage 5 |

Thresholds are computed **per VOD**: 90th percentile of the RMS envelope for
spikes, 25th (floored at 0.015) for the noise floor. `COOLDOWN_SECONDS` keeps
two spikes that are close together from both firing.

Also note `end` extension: the segment grows up to `MAX_TAIL_SECONDS` (6 s)
while the audio stays above the noise floor, so a clip does not cut off the
payoff. The last word's `end` is pushed along with it, otherwise subtitles
vanish before the audio does.

---

## `scored_segments.json` — written by `score_segments.py`

**Only the highlights survive this stage** — the array no longer contains every
segment.

```jsonc
[
  {
    "start": 12.34,
    "end": 20.15,
    "text": "allora ragazzi buonasera a nanna ENERGY SPIKE!",
    "words": [ ... ],
    "energy_spikes": [13.2],
    "cinematic_spike": true,
    "audio_multiplier": 3.41,
    "score": 38.117,          // NUOVO
    "god_tier": true          // NUOVO
  }
]
```

The score combines weighted trigger categories, brevity bias, repetition and
punctuation, then is multiplied by `audio_multiplier`.

Two things worth knowing:

- **Text is rewritten here.** The energy marker becomes a human-readable label
  and `slang_fixes` from `vocabulary.json` repair Whisper's mishearing of your
  catchphrases. This rewritten text is what ends up in filenames and reports.
- **The blacklist zeroes a segment** (`score = 0.1`) when it matches
  meta-streaming talk — ads, bitrate, network problems. It is a prefix match on
  normalised text, so entries are **accent-sensitive**: `pubblicità` matches the
  accented spelling only.

`god_tier` bypasses the minimum-duration filter in the next stage.

---

## `qa_validator.py` — no artifact, may rewrite the input

`In place`. If two `cinematic_spike` segments fall within `COOLDOWN_SECONDS` of
each other, the later one is demoted (`cinematic_spike: false`,
`audio_multiplier: 1.0`) and the file is rewritten. It exists because
stage 4's cooldown can be defeated by segments processed out of order.

---

## `final_choice.json` — written by `judge_agent.py`

```jsonc
[
  {
    "start": 9.84,             // allargato: padding, min 12 s, max 58 s
    "end": 27.5,
    "text": "...",
    "words": [ ... ],
    "energy_spikes": [13.2],
    "cinematic_spike": true,
    "audio_multiplier": 3.41,
    "metadata": {              // NUOVO
      "id": 7,
      "tension_score": 8.5,
      "hook": "the hook written by the LLM",
      "caption": "why this was picked",
      "viral_twist": "the expected payoff"
    }
  }
]
```

Segments are padded (`PADDING_START` / `PADDING_END`), floored at
`MIN_DURATION` (12 s) and capped at `MAX_DURATION_CEILING` (58 s). Overlapping
clips are rejected.

For very long candidates (> 55 s) **smart centering** kicks in: the cut is
anchored on the hype payoff, 30 s before and 15 s after, instead of padding
blindly.

Selection order: LLM picks → if fewer than 5 clips, **Spike Recovery** →
**Hype Detector**, capped by `MAX_HYPE_RECOVERY`. Without an API key only the
two fallbacks run.

Side effect: a `Telemetry_<date>.md` report is written next to the JSON, with
one section per clip and a tick-box row for TikTok / Shorts / Reels.

---

## `cut_engine.py` — written to `clips/`

No JSON. One `Clip_<n>_<slug>.mp4` per entry, 1080×1920.

Rendering is two passes: encode the vertical stack uncut, then burn in
subtitles with a single re-encode. The video filter is built by
`config.ffmpeg_vertical_stack_filter()` (webcam cropped on top, game below,
`VIDEO_FPS`), audio by `config.ffmpeg_mix_audio_filter()` (mic and game mixed
with per-track volumes). Codec and timeout come from
`config.ffmpeg_video_encoder_args()` / `config.ffmpeg_timeout()`.

Subtitles use `SUB_WORD_SOURCE`:

- `clip` (default) — Whisper re-runs on each rendered clip, so word timings
  match the clip's own audio. One extra model load per run, shared across clips.
- `vod` — reuses `transcription.json`, offset by `start`. Faster, less accurate.

`manual_cutter.py` renders through the same path and writes to
`Clips_Manuali/`.

---

## Files in `output/` that are diagnostic only

`stable_ts_probe.json` comes from `stable_ts_probe.py`, an A/B experiment.
Nothing in the pipeline reads it.
