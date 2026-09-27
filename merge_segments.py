import json
import os
import re
import sys

import config
import vocabulary

INPUT_FILE = os.path.join(config.OUTPUT_DIR, "transcription.json")
OUTPUT_FILE = os.path.join(config.OUTPUT_DIR, "merged_segments.json")

# --- PARAMETRI COMPORTAMENTALI ULTRA-CHIRURGICI ---
BASE_MAX_GAP = 0.8          # Silenzio standard max per discorsi fluidi (ideale per sub snelli)
HYPE_MAX_GAP = 4.0          # Abbassato da 6.0 a 4.0 per evitare accumuli di testo ingestibili
PUNCTUATION_MAX_GAP = 0.4   # Se c'è ?.!, basta un'esitazione di 0.4s per staccare la frase
MAX_CLIP_DURATION = 42.0    # Tetto massimo ideale per lo Short prima dello scoring

# Frasi e parole clou: arrivano dal vocabulary, sono il gusto di chi streamma e
# non una proprietà del codice. Match a parola intera o frase (no substring
# "go" in "prego"). Se vocabulary.json manca le liste sono vuote e il merge
# lavora solo sui silenzi e sulla punteggiatura.
HYPE_PHRASES = vocabulary.hype_phrases()
HYPE_WORDS_EXACT = vocabulary.hype_words()


def _norm_tokens(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())


def segment_has_hype(text: str) -> bool:
    norm = _norm_tokens(text)
    # Il marker del picco energetico è un segnale interno della pipeline: vale
    # come hype anche se non è una frase detta dallo streamer. _norm_tokens
    # toglie le parentesi quadre, quindi il confronto è sul token normalizzato.
    if config.ENERGY_SPIKE_TOKEN in norm:
        return True
    if any(p in norm for p in HYPE_PHRASES):
        return True
    return any(w in HYPE_WORDS_EXACT for w in norm.split())


def clean_text(text):
    return " ".join(text.split()).strip()

def merge_whisper_segments():
    print("🧩 MERGE ENGINE V4 (FIXED): Punctuation Guard & Adaptive Subtitle Splitter...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Errore: {INPUT_FILE} non trovato.")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_segments = data.get("segments", [])
    if not raw_segments:
        print("⚠️ Nessun segmento trovato nel file di trascrizione.")
        sys.exit(1)

    merged_segments = []
    
    # Inizializzazione primo blocco
    current = {
        "start": raw_segments[0]["start"],
        "end": raw_segments[0]["end"],
        "text": raw_segments[0]["text"],
        "words": list(raw_segments[0].get("words", [])),
        "energy_spikes": list(raw_segments[0].get("energy_spikes", [])),
    }

    for next_seg in raw_segments[1:]:
        prev_text_lower = current["text"].lower()
        next_text_lower = next_seg["text"].lower()
        
        has_hype_prev = segment_has_hype(prev_text_lower)
        has_hype_next = segment_has_hype(next_text_lower)
        hype_bridge = has_hype_prev or has_hype_next

        allowed_gap = HYPE_MAX_GAP if hype_bridge else BASE_MAX_GAP
        
        # Calcolo dei timestamp reali basati sulle parole
        real_current_end = current["end"]
        if current.get("words"):
            real_current_end = current["words"][-1]["end"]
            
        real_next_start = next_seg["start"]
        if next_seg.get("words"):
            real_next_start = next_seg["words"][0]["start"]

        gap = real_next_start - real_current_end
        potential_duration = next_seg["end"] - current["start"]

        # 🎯 PUNCTUATION GUARD: Verifica se la frase precedente è logicamente conclusa
        ends_with_punctuation = bool(re.search(r'[?.!]\s*$', current["text"]))

        # Punctuation guard: non spezzare raffiche hype anche se finiscono con !/? 
        if ends_with_punctuation and gap > PUNCTUATION_MAX_GAP and not hype_bridge:
            is_too_long_silence = True
        elif gap > allowed_gap:
            is_too_long_silence = True
        else:
            is_too_long_silence = False

        # Condizione di unione ottimizzata
        if gap <= allowed_gap and potential_duration <= MAX_CLIP_DURATION and not is_too_long_silence:
            current["end"] = next_seg["end"]
            current["text"] += " " + next_seg["text"]
            if "words" in next_seg:
                current["words"].extend(next_seg["words"])
            if next_seg.get("energy_spikes"):
                current.setdefault("energy_spikes", []).extend(next_seg["energy_spikes"])
        else:
            current["text"] = clean_text(current["text"])
            merged_segments.append(current)
            current = {
                "start": next_seg["start"],
                "end": next_seg["end"],
                "text": next_seg["text"],
                "words": list(next_seg.get("words", [])),
                "energy_spikes": list(next_seg.get("energy_spikes", [])),
            }

    # Inserimento ultimo blocco rimasto appeso
    current["text"] = clean_text(current["text"])
    merged_segments.append(current)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_segments, f, indent=2, ensure_ascii=False)

    print(f"✅ Compattazione completata. Generati {len(merged_segments)} segmenti puliti su {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_whisper_segments()