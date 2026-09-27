import json
import re
import os
import sys

import config
import vocabulary

# ==============================
# CONFIGURAZIONE INPUT/OUTPUT
# ==============================
INPUT_FILE = os.path.join(config.OUTPUT_DIR, "segments_with_audio.json")
OUTPUT_FILE = os.path.join(config.OUTPUT_DIR, "scored_segments.json")

# ==============================
# PARAMETRI SCORING & MERGE
# ==============================
TRIGGER_WEIGHT = 2.0
INTENSITY_WEIGHT = 1.0 # Peso abbassato per non premiare i logorroici
REPETITION_WEIGHT = 1.2
PUNCTUATION_WEIGHT = 2.5 # Bonus per stupore e domande
GOD_TIER_MULTIPLIER = 2.5

TOP_PERCENTILE = 0.30       
MERGE_GAP = 2.0             
MIN_DURATION = 4.0          

# ==============================
# KEYWORDS & BLACKLIST NLP
# ==============================
# Blacklist più precisa — solo meta-streaming puro: parlare di pubblicità, di
# bitrate o di problemi di rete non è mai un momento da clip.
TOXIC_META_WORDS = [
    "twitch chat", "pubblicità", "bitrate",
    "problema di rete", "stream scatta"  # frasi intere, non parole singole
]

# Trigger: le parole che danno punteggio. Le categorie e i loro pesi sono parte
# dell'algoritmo, le parole dentro sono il gusto di chi streamma e arrivano dal
# vocabulary. Se vocabulary.json manca, lo scoring resta su intensità,
# ripetizione, punteggiatura e segnale acustico.
TRIGGERS = vocabulary.triggers()

def apply_text_fixes(text):
    """Corregge le trascrizioni sbagliate di Whisper prima dello scoring e
    dell'invio all'AI, e rende leggibile il marker del picco energetico."""
    cleaned_text = re.sub(
        re.escape(config.ENERGY_SPIKE_MARKER),
        config.ENERGY_SPIKE_LABEL,
        text,
        flags=re.IGNORECASE,
    )
    for pattern, replacement in vocabulary.slang_fixes().items():
        # Sostituzione case-insensitive che mantiene la struttura della frase
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
    return cleaned_text

# ==============================
# UTILITY TEXT & SCORE
# ==============================
def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def count_triggers(text):
    score = 0
    god_tier_hit = False
    for category_name, category in TRIGGERS.items():
        for phrase in category:
            phrase_norm = normalize(phrase)
            if len(phrase_norm) <= 3:
                matched = bool(re.search(rf"\b{re.escape(phrase_norm)}\b", text))
            else:
                matched = phrase_norm in text
            if not matched:
                continue
            if category_name == "god_tier":
                score += GOD_TIER_MULTIPLIER
                god_tier_hit = True
            else:
                score += 1
    return score, god_tier_hit

def repetition_bonus(text):
    words = text.split()
    unique = set(words)
    if not words: return 0
    return 1 - (len(unique) / len(words))

def compute_score(segment):
    text_raw = segment["text"]
    text_norm = normalize(text_raw)
    duration = segment["end"] - segment["start"]
    words = text_norm.split()
    word_count = len(words)

    if duration <= 0 or word_count == 0:
        segment["god_tier"] = False
        return 0

    # 1. Filtro Blacklist (Penalità letale per il meta-streaming)
    for toxic in TOXIC_META_WORDS:
        if toxic in text_norm:
            return 0.1 # Shadowban istantaneo del segmento

    # 2. Base Scoring
    trigger_score, god_tier_hit = count_triggers(text_norm)
    rep_score = repetition_bonus(text_norm)
    
    # 3. Brevity & Punchline Bias (Meno parole in più tempo = Tensione)
    intensity_score = min(duration / max(word_count, 1), 5) 

    # 4. Punctuation Bonus (Stupore organico)
    punct_score = 0
    if "?" in text_raw or "!" in text_raw:
        punct_score = PUNCTUATION_WEIGHT

    base_score = (
        trigger_score * TRIGGER_WEIGHT +
        intensity_score * INTENSITY_WEIGHT +
        rep_score * REPETITION_WEIGHT +
        punct_score
    )

    # 🚀 FIX INGEGNERISTICO: Intercettiamo il picco audio RMS dal testo grezzo!
    # Bonus massiccio per forzare il segmento a superare la mannaia della soglia dinamica
    if config.ENERGY_SPIKE_LABEL in text_raw:
        base_score += config.ENERGY_SPIKE_SCORE_BONUS
        god_tier_hit = True  # Forziamo a True così bypassa il filtro di durata minima se la clip è corta!

    # 5. Cinema Multiplier
    if segment.get("cinematic_spike", False):
        final_score = base_score * segment.get("audio_multiplier", 1.0)
    else:
        final_score = base_score

    segment["god_tier"] = god_tier_hit
    return round(final_score, 3)

# ==============================
# MINI-MERGE & FILTERING
# ==============================
def auto_merge_tail_highlights(highlights):
    if not highlights:
        return []

    highlights = sorted(highlights, key=lambda x: x["start"])
    merged = [highlights[0].copy()]
    # Copia difensiva delle liste parole/spike (altrimenti si mutano i segmenti sorgente)
    if "words" in merged[0]:
        merged[0]["words"] = list(merged[0].get("words") or [])
    if "energy_spikes" in merged[0]:
        merged[0]["energy_spikes"] = list(merged[0].get("energy_spikes") or [])

    for nxt in highlights[1:]:
        current = merged[-1]
        gap = nxt["start"] - current["end"]
        if gap <= MERGE_GAP:
            current["end"] = max(current["end"], nxt["end"])
            current["text"] += " " + nxt["text"]
            current["score"] = max(current["score"], nxt["score"])
            current["god_tier"] = current.get("god_tier", False) or nxt.get("god_tier", False)
            # Conserva words per smart-centering judge / sub VOD
            if nxt.get("words"):
                current.setdefault("words", [])
                current["words"].extend(nxt["words"])
            if nxt.get("energy_spikes"):
                current.setdefault("energy_spikes", [])
                current["energy_spikes"].extend(nxt["energy_spikes"])
            # Propaga spike cinema se uno dei due lo ha
            if nxt.get("cinematic_spike"):
                current["cinematic_spike"] = True
                current["audio_multiplier"] = max(
                    float(current.get("audio_multiplier", 1.0)),
                    float(nxt.get("audio_multiplier", 1.0)),
                )
        else:
            copy_n = nxt.copy()
            if "words" in copy_n:
                copy_n["words"] = list(copy_n.get("words") or [])
            if "energy_spikes" in copy_n:
                copy_n["energy_spikes"] = list(copy_n.get("energy_spikes") or [])
            merged.append(copy_n)

    return merged

def filter_short_highlights(highlights):
    filtered = []
    for h in highlights:
        if (h["end"] - h["start"]) >= MIN_DURATION or h.get("god_tier", False):
            filtered.append(h)
    return filtered

# ==============================
# MAIN ENGINE
# ==============================
def score_segments():
    print(f"📊 SCORING ENGINE V7: Lore & Tension Bias (Anti-Meta Streaming)")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        segments = json.load(f)

    for seg in segments:
        # Pulisce il testo prima di calcolare il punteggio e di salvarlo
        seg["text"] = apply_text_fixes(seg["text"])
        seg["score"] = compute_score(seg)

    scores = [s["score"] for s in segments]
    if not scores:
        print("❌ Nessun segmento trovato.")
        sys.exit(1)

    scores_sorted = sorted(scores)
    index = min(len(scores_sorted) - 1, int(len(scores_sorted) * (1 - TOP_PERCENTILE)))
    threshold = scores_sorted[index]

    raw_highlights = [s for s in segments if s["score"] >= threshold]
    merged = auto_merge_tail_highlights(raw_highlights)
    highlights = filter_short_highlights(merged)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(highlights, f, indent=2, ensure_ascii=False)

    print(f"🔥 Segmenti analizzati: {len(segments)}")
    print(f"🎯 Threshold Dinamico (Top 30%): {round(threshold,3)}")
    print(f"🏆 Highlight grezzi: {len(raw_highlights)}")
    print(f"✂️ Highlight finali (Post-Filtro): {len(highlights)}")

if __name__ == "__main__":
    score_segments()