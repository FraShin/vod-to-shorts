"""
Sottotitoli clip: chunk 1–3 parole ravvicinate, timing Whisper + lag minimo.
- 1 parola → giallo fisso
- 2–3 parole → stile TikTok (attiva gialla, altre grigie) o tutte gialle (SUB_MULTI_WORD_STYLE)
"""
from __future__ import annotations

import copy
import re
from typing import Any

import config

MIN_CUE_DURATION = 0.14
MAX_WORD_ON_SCREEN_SEC = 0.38
CUE_TAIL_PAD_SEC = 0.05

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,95,&H0000FFFF,&H00666666,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,1,5,0,2,10,10,280,1
"""

ASS_FIXED_POS = r"{\an2\q2\pos(540,1720)}"
ASS_COLOR_YELLOW = r"{\c&H00FFFF&}"
ASS_COLOR_DIM = r"{\c&H666666&}"

ASS_EVENTS_HEAD = (
    "\n[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)

_STRONG_BREAK = re.compile(r"[.!?…][\"')\]]*$")


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# Token meta (marker del picco energetico, tag della pipeline) — mai a schermo.
# Il filtro si costruisce dal marker definito in config: così rinominarlo non
# lascia indietro questo file, e nessun token interno della pipeline finirà
# mai bruciato nel video come se fosse una parola detta dallo streamer.
def _build_meta_filter(marker: str):
    """Dal marker '[ENERGY_SPIKE]' ricava le forme che possono restare nel testo
    dopo la sanitizzazione: con o senza parentesi, separate da spazio/underscore/
    trattino, e i pezzi parziali concatenati (es. 'ENERGYSPIKE', 'ENERGY', 'SPIKE')."""
    parts = [p for p in re.split(r"[\s_/-]+", marker.strip("[]").strip()) if p]
    if not parts:
        # Marker vuoto: pattern che non matcha mai nulla.
        return re.compile(r"(?!x)x"), frozenset()

    joined = r"[\s_/-]*".join(re.escape(p) for p in parts)
    pattern = re.compile(rf"(?i)^\s*\[?\s*{joined}\s*\]?\s*$")

    compacts = set()
    for i in range(len(parts)):
        for j in range(i + 1, len(parts) + 1):
            compacts.add(re.sub(r"[^A-Z0-9]", "", "".join(parts[i:j]).upper()))
    return pattern, frozenset(compacts)


_META_WORD_RE, _META_COMPACT_VARIANTS = _build_meta_filter(config.ENERGY_SPIKE_MARKER)


def _sanitize_display_text(raw: str) -> str:
    text = raw.strip().upper()
    text = re.sub(r"[^a-zA-Z0-9!?ÀÈÉÌÒÙ']", "", text)
    return text


def _is_meta_word(raw: str) -> bool:
    """Parole fake da energy detector / tag scoring: escludi da ASS."""
    if not raw or not str(raw).strip():
        return True
    s = str(raw).strip()
    if _META_WORD_RE.match(s):
        return True
    # Dopo la sanitizzazione resta spesso la forma compattata (es. ENERGYSPIKE)
    compact = re.sub(r"[^a-zA-Z0-9]", "", s).upper()
    return compact in _META_COMPACT_VARIANTS


def words_for_clip(
    all_words: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
    pad_end: float = 0.15,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for w in all_words:
        ws, we = float(w["start"]), float(w["end"])
        if we < clip_start or ws > clip_end + pad_end:
            continue
        wc = copy.deepcopy(w)
        wc["start"] = max(ws, clip_start)
        wc["end"] = min(we, clip_end + pad_end)
        if wc["end"] <= wc["start"]:
            wc["end"] = wc["start"] + MIN_CUE_DURATION
        out.append(wc)
    return sorted(out, key=lambda x: float(x["start"]))


def _prepare_word_list(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for w in words:
        raw = w.get("word", "")
        if _is_meta_word(raw):
            continue
        display = _sanitize_display_text(raw)
        if not display or _is_meta_word(display):
            continue
        wc = copy.deepcopy(w)
        wc["display"] = display
        wc["raw"] = raw
        prepared.append(wc)
    return prepared


def _chunk_display_len(chunk: list[dict[str, Any]]) -> int:
    if not chunk:
        return 0
    return sum(len(str(w["display"])) for w in chunk) + len(chunk) - 1


def _chunk_would_overflow(
    current: list[dict[str, Any]],
    new_word: dict[str, Any],
) -> bool:
    """Evita righe troppo larghe (Montserrat 95 su 1080px)."""
    text = str(new_word["display"])
    max_chars = config.SUB_CHUNK_MAX_CHARS
    long_cut = config.SUB_CHUNK_LONG_WORD_CHARS

    if len(text) >= long_cut and current:
        return True
    extra = (1 if current else 0) + len(text)
    return _chunk_display_len(current) + extra > max_chars


def chunk_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Raggruppa fino a SUB_CHUNK_MAX_WORDS se ravvicinate, rispettando larghezza riga."""
    gap = config.SUB_CHUNK_GAP_SEC
    max_w = config.SUB_CHUNK_MAX_WORDS
    prepared = _prepare_word_list(words)
    if not prepared:
        return []

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for w in prepared:
        if current:
            prev = current[-1]
            pause = float(w["start"]) - float(prev["end"])
            prev_raw = str(prev.get("raw", ""))
            if (
                pause >= gap
                or len(current) >= max_w
                or _STRONG_BREAK.search(prev_raw)
                or _chunk_would_overflow(current, w)
            ):
                chunks.append(current)
                current = []
        current.append(w)

    if current:
        chunks.append(current)
    return chunks


def _chunk_line_text(tokens: list[str], active_index: int) -> str:
    style = config.SUB_MULTI_WORD_STYLE
    if len(tokens) == 1 or style == "all_yellow":
        return " ".join(tokens)
    parts: list[str] = []
    for j, tok in enumerate(tokens):
        if j == active_index:
            parts.append(f"{ASS_COLOR_YELLOW}{tok}")
        else:
            parts.append(f"{ASS_COLOR_DIM}{tok}")
    return " ".join(parts)


def _word_relative_times(
    w: dict[str, Any],
    clip_start: float,
    lag: float,
    prev_cue_end: float | None,
    next_in_chunk: dict[str, Any] | None,
    clip_duration: float | None,
) -> tuple[float, float]:
    rel_start = max(0.0, (float(w["start"]) - clip_start) + lag)
    rel_end = max(
        rel_start + MIN_CUE_DURATION,
        (float(w["end"]) - clip_start) + lag + CUE_TAIL_PAD_SEC,
    )
    rel_end = min(rel_end, rel_start + MAX_WORD_ON_SCREEN_SEC)

    if next_in_chunk is not None:
        nxt = (float(next_in_chunk["start"]) - clip_start) + lag
        rel_end = min(rel_end, max(rel_start + MIN_CUE_DURATION, nxt - 0.02))

    if prev_cue_end is not None and rel_start < prev_cue_end:
        rel_start = prev_cue_end + 0.02
        rel_end = max(rel_end, rel_start + MIN_CUE_DURATION)

    if clip_duration is not None:
        if rel_start >= clip_duration:
            return rel_start, rel_start
        rel_end = min(rel_end, clip_duration)

    return rel_start, rel_end


def build_ass_dialogues(
    words: list[dict[str, Any]],
    clip_start: float = 0.0,
    sub_delay: float = 0.0,
    clip_duration: float | None = None,
) -> list[tuple[float, float, str]]:
    lag = config.SUB_SYNC_LAG_SEC + sub_delay
    dialogues: list[tuple[float, float, str]] = []
    prev_end: float | None = None

    for chunk in chunk_words(words):
        tokens = [str(w["display"]) for w in chunk]
        if len(chunk) == 1:
            w = chunk[0]
            rs, re = _word_relative_times(
                w, clip_start, lag, prev_end, None, clip_duration
            )
            if clip_duration is not None and rs >= clip_duration:
                continue
            text = ASS_FIXED_POS + tokens[0]
            dialogues.append((rs, re, text))
            prev_end = re
            continue

        for i, w in enumerate(chunk):
            nxt = chunk[i + 1] if i + 1 < len(chunk) else None
            rs, re = _word_relative_times(
                w, clip_start, lag, prev_end, nxt, clip_duration
            )
            if clip_duration is not None and rs >= clip_duration:
                break
            line = ASS_FIXED_POS + _chunk_line_text(tokens, i)
            dialogues.append((rs, re, line))
            prev_end = re

    return dialogues


def write_ass(dialogues: list[tuple[float, float, str]], ass_path: str) -> None:
    events = ASS_EVENTS_HEAD
    for rs, re, text in dialogues:
        events += (
            f"Dialogue: 0,{format_timestamp(rs)},"
            f"{format_timestamp(re)},Default,,0,0,0,,{text}\n"
        )
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER + events)


def generate_ass_from_words(
    words: list[dict[str, Any]],
    ass_path: str,
    clip_start: float = 0.0,
    sub_delay: float = 0.0,
    clip_duration: float | None = None,
) -> None:
    dialogues = build_ass_dialogues(
        words, clip_start=clip_start, sub_delay=sub_delay, clip_duration=clip_duration
    )
    write_ass(dialogues, ass_path)


def generate_clip_ass(
    clip_start: float,
    clip_end: float,
    all_words: list[dict[str, Any]],
    ass_path: str,
    sub_delay: float = 0.0,
) -> None:
    """Legacy VOD path (manual_cutter). Preferire clip Whisper in cut_engine."""
    valid = words_for_clip(all_words, clip_start, clip_end)
    duration = clip_end - clip_start
    generate_ass_from_words(
        valid,
        ass_path,
        clip_start=clip_start,
        sub_delay=sub_delay,
        clip_duration=duration,
    )


def load_words_from_transcription(transcription_path: str) -> list[dict[str, Any]]:
    import json
    import os

    if not os.path.exists(transcription_path):
        return []
    with open(transcription_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_words: list[dict[str, Any]] = []
    for seg in data.get("segments", []):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        for w in seg.get("words", []):
            wc = copy.deepcopy(w)
            ws = max(float(wc["start"]), seg_start)
            we = min(float(wc["end"]), seg_end)
            if we <= ws:
                continue
            wc["start"] = ws
            wc["end"] = we
            all_words.append(wc)
    return sorted(all_words, key=lambda x: float(x["start"]))


def words_from_whisper_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Estrae parole da segmenti clip (timestamp 0 = inizio clip)."""
    out: list[dict[str, Any]] = []
    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        raw = seg.get("words") or []
        if not raw:
            continue
        for w in raw:
            wc = copy.deepcopy(w)
            ws = max(seg_start, float(wc["start"]))
            we = min(seg_end, float(wc["end"]))
            if we <= ws:
                we = ws + MIN_CUE_DURATION
            wc["start"] = ws
            wc["end"] = we
            out.append(wc)
    return sorted(out, key=lambda x: float(x["start"]))