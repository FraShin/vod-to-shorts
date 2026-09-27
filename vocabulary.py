"""
Gusto dello streamer: catch-phrase, trigger e correzioni.

Il codice di questa pipeline e' GENERICO. Tutto quello che dipende dalla persona
che streamma (cosa urla, come parla, che stile ha) vive in `vocabulary.json`,
accanto a questi script e fuori dal repository:

    vocabulary.example.json   modello vuoto, versionato su git
    vocabulary.json           i tuoi valori, gitignorato

Il file e' OPZIONALE. Se manca, tutte le liste restano vuote e la pipeline gira
lo stesso: rilevamento acustico (Cinema Engine), scoring di base e sottotitoli
funzionano comunque, semplicemente non riconosce le tue frasi clou.

Le chiavi che iniziano con `_` sono ignorate: servono per le note nel file.
"""
from __future__ import annotations

import json
import os
from typing import Any

import config

VOCABULARY_FILE = os.environ.get("VOCABULARY_FILE") or os.path.join(
    config.SCRIPTS_DIR, "vocabulary.json"
)

# Struttura attesa. Fa anche da valore di ripiego per ogni sezione mancante,
# cosi' un vocabulary.json parziale non fa mai esplodere la pipeline.
_DEFAULTS: dict[str, Any] = {
    "profile": {
        "name": "",
        "genres": [],
        "description": "",
        "selection_notes": [],
    },
    "whisper_prompt": {
        "vod": "",
        "clip": "",
    },
    "hype": {
        "phrases": [],
        "words": [],
        "absolute_phrases": [],
    },
    "triggers": {
        "god_tier": [],
        "shock": [],
        "panic": [],
        "clutch": [],
        "lore": [],
    },
    "slang_fixes": {},
}

_cache: dict[str, Any] | None = None


def _strip_notes(obj: Any) -> Any:
    """Rimuove ricorsivamente le chiavi di documentazione (che iniziano con '_')."""
    if isinstance(obj, dict):
        return {
            k: _strip_notes(v) for k, v in obj.items() if not str(k).startswith("_")
        }
    if isinstance(obj, list):
        return [_strip_notes(v) for v in obj]
    return obj


def _merge(base: dict, override: dict) -> dict:
    """Sovrappone `override` a `base`, sezione per sezione."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _clean_list(value: Any) -> list[str]:
    """Lista di stringhe, minuscole e senza spazi: il testo viene normalizzato
    cosi' prima del confronto, quindi accettiamo qualunque maiuscola l'utente
    abbia scritto a mano nel JSON."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip().lower()
        if text:
            out.append(text)
    return out


def load_vocabulary(path: str | None = None) -> dict:
    """Legge il vocabulary e lo fonde sui valori di ripiego. Non solleva mai."""
    path = path or VOCABULARY_FILE

    if not os.path.isfile(path):
        return _merge(_DEFAULTS, {})

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"⚠️  vocabulary.json illeggibile ({e}). Procedo con liste vuote.")
        return _merge(_DEFAULTS, {})

    if not isinstance(raw, dict):
        print("⚠️  vocabulary.json: la radice deve essere un oggetto JSON. Liste vuote.")
        return _merge(_DEFAULTS, {})

    data = _merge(_DEFAULTS, _strip_notes(raw))

    # Normalizzazione difensiva dei campi che il codice confronta in minuscolo.
    hype = data["hype"]
    for key in ("phrases", "words", "absolute_phrases"):
        hype[key] = _clean_list(hype.get(key))
    data["triggers"] = {
        category: _clean_list(values)
        for category, values in data["triggers"].items()
    }
    if not isinstance(data["slang_fixes"], dict):
        data["slang_fixes"] = {}

    return data


def vocabulary() -> dict:
    """Istanza condivisa: il file si legge una volta sola per processo."""
    global _cache
    if _cache is None:
        _cache = load_vocabulary()
    return _cache


# ==============================================================
# Accessori — usare questi, non `vocabulary()[...]`
# ==============================================================

def profile() -> dict:
    return vocabulary()["profile"]


def whisper_prompt(kind: str = "vod") -> str:
    """Prompt di contesto per Whisper ('vod' o 'clip'). Stringa vuota se assente."""
    prompts = vocabulary()["whisper_prompt"]
    value = prompts.get(kind) or prompts.get("vod") or ""
    return str(value).strip()


def hype_phrases() -> tuple[str, ...]:
    return tuple(vocabulary()["hype"]["phrases"])


def hype_words() -> frozenset[str]:
    return frozenset(vocabulary()["hype"]["words"])


def hype_absolute_phrases() -> tuple[str, ...]:
    return tuple(vocabulary()["hype"]["absolute_phrases"])


def triggers() -> dict[str, list[str]]:
    return vocabulary()["triggers"]


def slang_fixes() -> dict[str, str]:
    return vocabulary()["slang_fixes"]
