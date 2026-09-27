#!/usr/bin/env python3
"""
Smoke test della pipeline — gira in 2 secondi, senza VOD, GPU o CUDA.

Verifica gli stadi "puri" (merge -> score -> qa), il filtro dei sottotitoli e
il prompt dell'agente, con dati finti. Serve a rispondere a una domanda sola:
"ho rotto qualcosa?" — senza dover aspettare un VOD intero per scoprirlo.

Uso, dalla cartella scripts/:
    python tests/smoke_pipeline.py

Exit code 0 = tutto a posto, 1 = qualcosa si è rotto.

Nota: audio_engine, transcribe_engine, audio_analysis e gli stadi FFmpeg NON
sono coperti (richiedono ffmpeg, torch, soundfile e un video vero). Qui si
simula solo il CONTRATTO dei loro output, cioè i campi JSON che gli stadi
successivi si aspettano.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

# config legge OUTPUT_DIR all'import: va impostato prima di importarlo, quindi
# il test scrive in una cartella temporanea e non tocca output/ e clips/.
_WORKDIR = tempfile.mkdtemp(prefix="pipeline_smoke_")
os.environ["OUTPUT_DIR"] = _WORKDIR

import config  # noqa: E402
import vocabulary  # noqa: E402

# judge_agent importa la libreria openai, che il test non usa: uno stub evita
# di doverla installare solo per controllare il prompt.
_stub = types.ModuleType("openai")
_stub.OpenAI = type("OpenAI", (), {})
sys.modules.setdefault("openai", _stub)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)


def section(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end, "probability": 0.9}


def build_fake_transcription() -> dict:
    """Sei segmenti che coprono i casi che contano."""
    return {
        "source_vod": "TEST.mp4",
        "source_vod_path": "/tmp/TEST.mp4",
        "segments": [
            {   # parlato neutro, isolato
                "start": 0.0, "end": 2.0, "text": "allora ragazzi buonasera",
                "words": [word("allora", 0.0, 0.4), word("ragazzi", 0.4, 0.9),
                          word("buonasera", 0.9, 1.6)],
                "energy_spikes": [],
            },
            {   # frase clou riconosciuta dal vocabulary
                "start": 2.5, "end": 3.5, "text": "a nanna",
                "words": [word("a", 2.5, 2.6), word("nanna", 2.6, 3.2)],
                "energy_spikes": [],
            },
            {   # a 1.8s dal payoff: si unisce SOLO grazie al bridge hype
                "start": 5.0, "end": 7.0, "text": "che bello questo gioco",
                "words": [word("che", 5.0, 5.3), word("bello", 5.3, 5.8),
                          word("questo", 5.8, 6.2), word("gioco", 6.2, 6.7)],
                "energy_spikes": [],
            },
            {   # picco energetico: marker interno + bonus di punteggio
                "start": 20.0, "end": 22.0,
                "text": f"che paura {config.ENERGY_SPIKE_MARKER}",
                "words": [word("che", 20.0, 20.4), word("paura", 20.4, 21.0)],
                "energy_spikes": [20.5],
            },
            {   # meta-streaming: deve essere azzerato dalla blacklist
                "start": 40.0, "end": 43.0,
                "text": "sto parlando della pubblicità del canale",
                "words": [word("sto", 40.0, 40.3), word("parlando", 40.3, 41.0),
                          word("della", 41.0, 41.4), word("pubblicità", 41.4, 42.0),
                          word("del", 42.0, 42.2), word("canale", 42.2, 42.6)],
                "energy_spikes": [],
            },
            {   # seconda frase clou del vocabulary, isolata
                "start": 60.0, "end": 64.0, "text": "porco trollo che roba",
                "words": [word("porco", 60.0, 60.5), word("trollo", 60.5, 61.1),
                          word("che", 61.1, 61.4), word("roba", 61.4, 61.9)],
                "energy_spikes": [],
            },
        ],
    }


def main() -> int:
    print(f"cartella di lavoro temporanea: {_WORKDIR}")
    print(f"vocabulary: {vocabulary.VOCABULARY_FILE}")
    if not os.path.isfile(vocabulary.VOCABULARY_FILE):
        print("  (vocabulary.json assente: liste vuote, il test controlla solo la meccanica)")

    fake = build_fake_transcription()
    with open(os.path.join(_WORKDIR, "transcription.json"), "w", encoding="utf-8") as f:
        json.dump(fake, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- config
    section("1. CONFIG")
    print(f"  OUTPUT_DIR   : {config.OUTPUT_DIR}")
    print(f"  marker       : {config.ENERGY_SPIKE_MARKER} / token '{config.ENERGY_SPIKE_TOKEN}'")
    print(f"  lingua       : {config.WHISPER_LANGUAGE} | modello {config.WHISPER_MODEL}")
    print(f"  clip target  : {config.TARGET_TOTAL_CLIPS}")
    print(f"  codec        : {' '.join(config.ffmpeg_video_encoder_args())}")
    check(config.OUTPUT_DIR == _WORKDIR, "OUTPUT_DIR sovrascrivibile via ambiente")
    check(config.ENERGY_SPIKE_TOKEN ==
          config.ENERGY_SPIKE_MARKER.strip("[]").lower().replace(" ", "_"),
          "il token del marker deriva dal marker (una sola fonte)")
    check("libx264" in " ".join(config.ffmpeg_video_encoder_args()) or
          "nvenc" in " ".join(config.ffmpeg_video_encoder_args()),
          "argomenti del codec coerenti")

    # ---------------------------------------------------------------- merge
    section("2. MERGE — bridge sulle frasi clou")
    import merge_segments

    merge_segments.merge_whisper_segments()
    with open(os.path.join(_WORKDIR, "merged_segments.json"), encoding="utf-8") as f:
        merged = json.load(f)
    for i, seg in enumerate(merged, 1):
        print(f"    [{i}] {seg['start']:6.1f}-{seg['end']:6.1f}  {seg['text']}")

    check(len(merged) == 4, f"6 segmenti -> 4 blocchi (trovati {len(merged)})")
    bridge = next((s for s in merged if "nanna" in s["text"].lower()), None)
    check(bridge is not None, "il blocco con la frase clou esiste")
    if bridge:
        check("gioco" in bridge["text"], "il seguito del payoff viene agganciato")
        check("buonasera" in bridge["text"], "anche il setup prima del payoff viene agganciato")

    # ------------------------------------------------- simulazione audio_analysis
    section("3. CONTRATTO DI audio_analysis (simulato)")
    for seg in merged:
        seg["cinematic_spike"] = False
        seg["audio_multiplier"] = 1.0
    spike_raw = next((s for s in merged if config.ENERGY_SPIKE_MARKER in s["text"]), None)
    check(spike_raw is not None, "il segmento col marker del picco esiste")
    if spike_raw:
        spike_raw["cinematic_spike"] = True
        spike_raw["audio_multiplier"] = 3.0
    with open(os.path.join(_WORKDIR, "segments_with_audio.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print("  (audio_analysis vero richiede numpy+soundfile+video.wav: qui se ne simula l'uscita)")

    # ---------------------------------------------------------------- score
    section("4. SCORE — trigger, bonus picco, blacklist")
    import score_segments

    score_segments.score_segments()
    with open(os.path.join(_WORKDIR, "scored_segments.json"), encoding="utf-8") as f:
        scored = json.load(f)
    for seg in scored:
        print(f"    score={seg['score']:7.2f}  god_tier={str(seg['god_tier']):5}  {seg['text'][:44]}")

    texts = " | ".join(s["text"] for s in scored)
    check(config.ENERGY_SPIKE_MARKER not in texts, "il marker grezzo non finisce nel testo finale")
    check(config.ENERGY_SPIKE_LABEL in texts,
          f"il marker diventa leggibile ({config.ENERGY_SPIKE_LABEL})")

    spike_seg = next((s for s in scored if config.ENERGY_SPIKE_LABEL in s["text"]), None)
    check(spike_seg is not None, "il segmento col picco sopravvive allo scoring")
    if spike_seg:
        check(spike_seg["god_tier"] is True, "il picco forza god_tier")
        soglia = config.ENERGY_SPIKE_SCORE_BONUS * 3.0
        check(spike_seg["score"] > soglia,
              f"bonus picco x moltiplicatore cinema ({spike_seg['score']} > {soglia})")

    # Blacklist: il confronto avviene su testo che NON perde gli accenti, quindi
    # la voce "pubblicità" scatta solo nella forma accentata.
    acc = score_segments.compute_score(
        {"text": "sto parlando della pubblicità del canale", "start": 40.0, "end": 43.0})
    noacc = score_segments.compute_score(
        {"text": "sto parlando della pubblicita del canale", "start": 40.0, "end": 43.0})
    print(f"    blacklist 'pubblicità': accento {acc} | senza accento {noacc}")
    check(acc == 0.1, "blacklist del meta-streaming attiva")

    # ---------------------------------------------------------------- qa
    section("5. QA VALIDATOR")
    import qa_validator

    check(qa_validator.validate_and_sanitize() is True, "QA passa su dati coerenti")

    # ---------------------------------------------------------------- sub
    section("6. SOTTOTITOLI — il marker non finisce mai a schermo")
    import subtitles

    for token in (config.ENERGY_SPIKE_MARKER, config.ENERGY_SPIKE_TOKEN,
                  f" {config.ENERGY_SPIKE_MARKER} ", "ENERGYSPIKE", "ENERGY", "SPIKE"):
        check(subtitles._is_meta_word(token), f"filtrato: {token!r}")
    for token in ("nanna", "ciao", "BOOM"):
        check(not subtitles._is_meta_word(token), f"parola vera lasciata passare: {token!r}")

    # ---------------------------------------------------------------- judge
    section("7. JUDGE — prompt dal profilo")
    try:
        import judge_agent

        prompt = judge_agent.build_system_prompt()
        prof = vocabulary.profile()
        check(bool(prompt.strip()), "il prompt viene costruito")
        check(str(judge_agent.TARGET_TOTAL_CLIPS) in prompt,
              "il numero di clip entra nel prompt")
        if prof.get("name"):
            check(prof["name"] in prompt, "il nome del profilo entra nel prompt")
        for genre in prof.get("genres") or []:
            check(genre in prompt, f"il genere '{genre}' entra nel prompt")

        cases = [
            '{"clips": [{"id": 1}]}',
            'Ecco:\n{"clips": [{"id": 1}]}\nSpero vada bene',
            "nessun json qui",
            None,
            "",
        ]
        for text in cases:
            judge_agent.safe_json_parse(text)
        check(judge_agent.safe_json_parse('{"clips": [{"id": 1}]}') == {"clips": [{"id": 1}]},
              "safe_json_parse: json pulito")
        check(judge_agent.safe_json_parse("spazzatura") is None,
              "safe_json_parse: spazzatura -> None senza eccezioni")
    except ImportError as e:
        print(f"  saltato: {e}")

    # ---------------------------------------------------------------- esito
    section("ESITO")
    if FAILURES:
        print(f"  {len(FAILURES)} controlli falliti:")
        for f in FAILURES:
            print(f"    - {f}")
        return 1
    print("  tutti i controlli passati")
    return 0


if __name__ == "__main__":
    sys.exit(main())
