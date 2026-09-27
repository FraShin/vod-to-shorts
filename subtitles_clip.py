"""Burn-in da Whisper sulla clip (delega a subtitles.py)."""
from __future__ import annotations

from typing import Any

from subtitles import generate_ass_from_words, words_from_whisper_segments


def generate_ass_clip_local(
    clip_duration: float,
    segments: list[dict[str, Any]],
    ass_path: str,
) -> None:
    words = words_from_whisper_segments(segments)
    generate_ass_from_words(
        words,
        ass_path,
        clip_start=0.0,
        sub_delay=0.0,
        clip_duration=clip_duration,
    )