#!/usr/bin/env python3
"""
Prova A/B stable-ts su video.wav (non sostituisce la pipeline).

Uso, dalla root del progetto:
  pip install -U stable-ts
  .venv/bin/python scripts/stable_ts_probe.py

Scrive output/stable_ts_probe.json con segmenti+words.
Poi confronta con output/transcription.json sulla stessa frase.
"""
from __future__ import annotations

import json
import os
import sys

import config

WAV = os.path.join(config.OUTPUT_DIR, "video.wav")
OUT = os.path.join(config.OUTPUT_DIR, "stable_ts_probe.json")


def main() -> int:
    if not os.path.isfile(WAV):
        print(f"❌ Manca {WAV} — lancia prima audio_engine.")
        return 1

    try:
        from stable_whisper import load_model
    except ImportError:
        print("❌ pip install -U stable-ts")
        return 1

    print("⏳ stable-ts large-v3 (primo run ~3 GB download)...")
    model = load_model(config.WHISPER_MODEL, device="cuda")
    result = model.transcribe(
        WAV,
        language=config.WHISPER_LANGUAGE,
        word_timestamps=True,
        vad=True,
        regroup=True,
    )

    segments = []
    for seg in result.segments:
        words = []
        for w in getattr(seg, "words", []) or []:
            words.append(
                {
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                }
            )
        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
                "words": words,
            }
        )

    payload = {
        "source_vod": os.path.basename(config.VIDEO_FILENAME or ""),
        "engine": "stable-ts",
        "segments": segments,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Salvato {OUT} ({len(segments)} segmenti)")
    return 0


if __name__ == "__main__":
    sys.exit(main())