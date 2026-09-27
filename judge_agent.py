from openai import OpenAI
import json
import os
import datetime
import sys
import re

import config
import vocabulary

INPUT_SCORED = os.path.join(config.OUTPUT_DIR, "scored_segments.json")
OUTPUT_SELECTION = os.path.join(config.OUTPUT_DIR, "final_choice.json")

MAX_CANDIDATES = 40
# Quante clip produrre per run — da .env / config (TARGET_TOTAL_CLIPS)
TARGET_TOTAL_CLIPS = config.TARGET_TOTAL_CLIPS

# Parametri scattanti per massimizzare la ritenzione nei primi 3 secondi
PADDING_START = 3.5
PADDING_END = 4.0
MIN_DURATION = 12.0
MAX_DURATION_CEILING = 58.0  # Tetto massimo per i video verticali

# Loop 3️⃣ Hype recovery — limiti per non sovrastare l'AI
MAX_HYPE_RECOVERY = 2
HYPE_RECOVERY_SCORE = 7.5  # sotto spike recovery (8.5), sopra clip medie

# Ancore dello smart-centering: frasi multi-parola e token singoli, dal
# vocabulary. Match esatto e non sottostringa, così "go" non scatta dentro
# "prego". Se il vocabulary è vuoto, il centering si basa solo sul segnale
# acustico e sulla durata.
HYPE_PHRASES_ANCHOR = vocabulary.hype_phrases()
HYPE_WORDS_EXACT = vocabulary.hype_words()

# Pattern rari per il recupero degli orfani: il sottoinsieme "iconico" del
# vocabulary. Tenuto separato perché qui il cap è basso e non vogliamo che una
# parola comune (es. "eliminato", frequentissimo in horror e soulslike) peschi
# a caso: deve scattare solo sul payoff davvero riconoscibile.
ABSOLUTE_HYPE_PHRASES = vocabulary.hype_absolute_phrases()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())


def find_hype_anchor_time(words: list) -> float | None:
    """Trova il payoff hype nel segmento (dalla fine verso l'inizio)."""
    if not words:
        return None

    joined = _normalize_text(" ".join(w.get("word", "") for w in words))
    for phrase in HYPE_PHRASES_ANCHOR:
        if phrase in joined:
            # Ancora all'ultima parola del segmento che fa parte della frase trovata
            phrase_tokens = phrase.split()
            for w in reversed(words):
                wc = _normalize_text(w.get("word", ""))
                if wc in phrase_tokens or wc == phrase_tokens[-1]:
                    return float(w["start"])

    for w in reversed(words):
        wc = _normalize_text(w.get("word", ""))
        # Il marker del picco energetico vale come ancora: è il segnale che il
        # Cinema Engine ha trovato un urlo, anche senza una frase riconosciuta.
        if wc in HYPE_WORDS_EXACT or wc == config.ENERGY_SPIKE_TOKEN:
            return float(w["start"])
    return None


def expand_clip_duration(clip):
    base_duration = clip["end"] - clip["start"]
    
    # --- SMART CENTERING PER SEGMENTI MOSTRO (Whisper Gaps) ---
    if base_duration > 55.0 and "words" in clip and clip["words"]:
        target_time = find_hype_anchor_time(clip["words"])

        if target_time is not None:
            clip["start"] = max(0, target_time - 30.0)
            clip["end"] = min(clip["end"], target_time + 15.0)
            return clip
            
    # --- LOGICA STANDARD PER CLIP CORTE O REGOLARI ---
    if base_duration >= 50.0:
        if base_duration > MAX_DURATION_CEILING:
            clip["end"] = clip["start"] + MAX_DURATION_CEILING
        return clip

    clip["start"] = max(0, clip["start"] - PADDING_START)
    clip["end"] = clip["end"] + PADDING_END
    duration = clip["end"] - clip["start"]

    if duration < MIN_DURATION:
        missing = MIN_DURATION - duration
        clip["start"] = max(0, clip["start"] - (missing / 2))
        clip["end"] += (missing / 2)
        
    if (clip["end"] - clip["start"]) > MAX_DURATION_CEILING:
        clip["end"] = clip["start"] + MAX_DURATION_CEILING

    return clip

def is_overlapping(new_clip, existing_clips):
    for existing in existing_clips:
        if new_clip["start"] < existing["end"] and new_clip["end"] > existing["start"]:
            return True
    return False

def safe_json_parse(response_text):
    """L'API può restituire testo attorno al JSON: si prova il parse diretto e
    poi si ritaglia il primo oggetto. Un formato inatteso non deve far cadere
    la pipeline, quindi si torna None e si prosegue con i fallback."""
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            return json.loads(response_text[start:end])
        except (json.JSONDecodeError, TypeError):
            return None

def generate_telemetry_md(clips, output_file_path):
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        dir_name = os.path.dirname(output_file_path)
        md_filename = os.path.join(dir_name, f"Telemetry_{today}.md")
        
        lines = [
            "---",
            f"vod_date: {today}",
            f"total_clips: {len(clips)}",
            "strategy: Performance_Max",
            "---\n",
            f"# Report VOD Telemetry: {today}\n"
        ]
        
        for i, clip in enumerate(clips):
            start = round(clip.get("start", 0), 1)
            end = round(clip.get("end", 0), 1)
            duration = round(end - start, 1)
            text = clip.get("text", "").replace("\n", " ").strip()
            meta = clip.get("metadata", {})
            score = meta.get("tension_score", "N/D")
            
            fallback_hook = " ".join(text.split()[:8]) + "..." if len(text.split()) > 8 else text
            clip_hook = meta.get("hook", fallback_hook)
            
            lines.append(f"## 🎬 Clip {i+1}: {clip_hook[:25]}...")
            lines.append(f"* **Start-End**: {start}s - {end}s")
            lines.append(f"* **Duration**: {duration}s")
            lines.append(f"* **Tension Score**: {score}")
            lines.append(f"* **Hook**: \"{clip_hook}\"")
            lines.append("* **Performance Status**: [ ] TikTok | [ ] Shorts | [ ] Reels\n")
            lines.append("**Trascrizione Completa**:")
            lines.append(f"> \"{text}\"\n")
            lines.append("---\n")
            
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✅ Telemetria reale generata: {md_filename}")
    except Exception as e:
        print(f"⚠️ Impossibile generare la telemetria: {e}")

def build_system_prompt() -> str:
    """Prompt dell'agente che sceglie le clip.

    L'ossatura (compito, formato, regole di base) è generica e vale per
    qualunque streamer; identità, generi e regole extra arrivano dal vocabulary.
    Così lo stesso codice seleziona clip per un horror e per uno di cucina.
    """
    prof = vocabulary.profile()
    name = str(prof.get("name") or "").strip()
    genres = [str(g).strip() for g in (prof.get("genres") or []) if str(g).strip()]
    description = str(prof.get("description") or "").strip()
    notes = [str(n).strip() for n in (prof.get("selection_notes") or []) if str(n).strip()]

    who = (
        f"Sei il Content Strategist personale di {name}."
        if name
        else "Sei un Content Strategist esperto di clip virali."
    )
    if genres:
        who += f" Streamer di {', '.join(genres)}."
    if description:
        who += f" Il tuo stile è {description}."

    rules = [
        "PRIORITÀ ASSOLUTA ai payoff emotivi e ai momenti reali di gioco: urli di vittoria, ganci sonori pesanti, rage improvvisi, morti del boss.",
        "GENERA TITOLI (HOOKS) UNICI: ogni clip deve avere un titolo gancio personalizzato basato sul testo. NON usare mai titoli generici o ripetuti.",
        "ELIMINA i menu, la gestione inventario, la lore passiva e le discussioni generiche che non hanno un payoff.",
        "CERCA l'effetto \"Fermi tutti\": clip che terminano con una battuta forte o uno shock.",
    ]

    # Gli esempi di payoff vengono dalle frasi clou dello streamer: è il modo
    # per dargli un riferimento concreto senza scriverlo nel codice.
    examples = [p for p in HYPE_PHRASES_ANCHOR if " " in p or len(p) > 3][:3]
    if examples:
        rules[0] += " Esempi di payoff tipici: " + ", ".join(f'"{e}"' for e in examples) + "."

    rules.extend(notes)
    numbered = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, start=1))

    return (
        f"{who}\n"
        f"Il tuo compito è spietato: selezionare le clip più virali. "
        f"Massimo {TARGET_TOTAL_CLIPS} elementi.\n\n"
        f"REGOLE DI SELEZIONE RIGIDE:\n{numbered}\n"
    )


def main():
    print("🚀 JUDGE AGENT V14 — Unified Score Mode")

    if not os.path.exists(INPUT_SCORED):
        print(f"❌ scored_segments.json non trovato in {INPUT_SCORED}")
        return False

    with open(INPUT_SCORED, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = [s for s in data if (s.get("score", 0) > 0.5 or s.get("cinematic_spike", False))]

    candidates.sort(
        key=lambda x: (x.get("score", 0), x.get("cinematic_spike", False), len(x.get("text", ""))),
        reverse=True
    )

    top_candidates = candidates[:MAX_CANDIDATES]

    candidates_text = ""
    for i, seg in enumerate(top_candidates):
        duration = round(seg["end"] - seg["start"], 2)
        spike_tag = "[SPIKE]" if seg.get('cinematic_spike', False) else ""
        candidates_text += f"ID: {i} {spike_tag}\nTesto: {seg['text']}\nDurata: {duration}s\nScore: {seg.get('score', 0):.2f}\n\n"

    api_key = (config.DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")).strip()

    client = None
    if api_key:
        client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            timeout=60.0,
        )
    else:
        print("⚠️ DEEPSEEK_API_KEY assente: solo Spike Recovery + Hype Detector.")

    system_prompt = build_system_prompt()

    user_prompt = (
        f"Analizza la lista e restituisci il json finale con le "
        f"{TARGET_TOTAL_CLIPS} clip migliori.\n\nLista:\n{candidates_text}"
    )
    clips_output = []

    try:
        if client:
            completion = client.chat.completions.create(
                model=getattr(config, "DEEPSEEK_MODEL", "deepseek-v4-flash"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            response = completion.choices[0].message.content
            parsed = safe_json_parse(response)
            if parsed and "clips" in parsed:
                clips_output = parsed["clips"][:TARGET_TOTAL_CLIPS]
                print(
                    f"🤖 DeepSeek ha risposto correttamente! Selezionate {len(clips_output)} clip virali intelligenti."
                )
    except Exception as e:
        print(f"❌ Errore API DeepSeek: {e}")

    final_clips = []

    # 1️⃣ INIEZIONE SCELTE INTELLIGENTI AI
    for clip in clips_output:
        if len(final_clips) >= TARGET_TOTAL_CLIPS:
            break
        idx = clip.get("id")
        if idx is None or idx >= len(top_candidates):
            continue

        base = top_candidates[idx].copy()
        expanded = expand_clip_duration(base)

        if not is_overlapping(expanded, final_clips):
            expanded["metadata"] = clip
            final_clips.append(expanded)

    # 2️⃣ SPIKE RECOVERY
    if len(final_clips) < 5:
        print("⚠️ Selezione AI insufficiente. Attivazione Spike Recovery...")
        for i, base in enumerate(top_candidates):
            if len(final_clips) >= TARGET_TOTAL_CLIPS:
                break
            if not base.get("cinematic_spike", False):
                continue
            if any(fc.get("metadata", {}).get("id") == i for fc in final_clips):
                continue
            if base.get("score", 0) < 0.75:
                continue

            test = expand_clip_duration(base.copy())
            if not is_overlapping(test, final_clips):
                words_sample = " ".join(base.get("text", "").split()[:5]) + "..."
                test["metadata"] = {
                    "id": i,
                    "tension_score": 8.5,
                    "hook": f"SPIKE: {words_sample}",
                    "caption": "Picco audio rilevato dal Cinema Engine",
                    "viral_twist": "Climax acustico recuperato."
                }
                final_clips.append(test)

    # 3️⃣ Recupero highlight orfani (pattern rari, cap e score sotto l'AI top)
    print("🎯 [HYPE DETECTOR] Analisi di recupero su segmenti orfani...")
    hype_added = 0
    for seg in data:
        if hype_added >= MAX_HYPE_RECOVERY:
            break
        if len(final_clips) >= TARGET_TOTAL_CLIPS:
            break

        text_normalized = _normalize_text(seg.get("text", ""))
        is_absolute_hype = any(pat in text_normalized for pat in ABSOLUTE_HYPE_PHRASES)

        if not is_absolute_hype:
            continue

        already_covered = any(fc["start"] <= seg["start"] <= fc["end"] for fc in final_clips)
        if already_covered:
            continue

        target_clip = expand_clip_duration(seg.copy())
        if is_overlapping(target_clip, final_clips):
            continue

        words_sample = " ".join(seg.get("text", "").split()[:5]) + "..."
        target_clip["metadata"] = {
            "id": 999,
            "tension_score": HYPE_RECOVERY_SCORE,
            "hook": f"CLIMAX: {words_sample}",
            "caption": "Recupero Hype Detector (pattern raro)",
            "viral_twist": "Payoff iconico recuperato.",
        }
        final_clips.append(target_clip)
        hype_added += 1
        print(
            f"👑 Hype recuperato al secondo {seg['start']} (score {HYPE_RECOVERY_SCORE}, {hype_added}/{MAX_HYPE_RECOVERY})"
        )

    # Ordinamento finale basato sullo score dei metadati
    final_clips.sort(key=lambda x: x["metadata"].get("tension_score", 5), reverse=True)
    final_clips = final_clips[:TARGET_TOTAL_CLIPS]

    with open(OUTPUT_SELECTION, "w", encoding="utf-8") as f:
        json.dump(final_clips, f, indent=2, ensure_ascii=False)

    print(f"💾 ARCHIVIO COMPATTATO: Salvate {len(final_clips)} clip reali in {OUTPUT_SELECTION}")
    generate_telemetry_md(final_clips, OUTPUT_SELECTION)
    if not final_clips:
        return False
    return True

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)