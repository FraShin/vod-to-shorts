import numpy as np
import soundfile as sf
import json
import os
import sys
import config

# ==============================
# CONFIGURAZIONE FILE & PARAMETRI
# ==============================
AUDIO_FILE = os.path.join(config.OUTPUT_DIR, "video.wav")
SEGMENTS_FILE = os.path.join(config.OUTPUT_DIR, "merged_segments.json")
OUTPUT_FILE = os.path.join(config.OUTPUT_DIR, "segments_with_audio.json")

WINDOW_SIZE = 0.2          # finestra RMS 200ms
MIN_SILENCE_SECONDS = 0.5  # durata minima di silenzio per spike cinematica
SILENCE_WINDOWS = int(MIN_SILENCE_SECONDS / WINDOW_SIZE)

MAX_MULTIPLIER = 4.0       # cap per moltiplicatore audio
ENERGY_TAIL_FACTOR = 1.2   # fattore rispetto al noise floor per tail
MAX_TAIL_SECONDS = 6.0     # max estensione tail energetico
COOLDOWN_SECONDS = 4.0     # cooldown per evitare frammentazione spike
COOLDOWN_WINDOWS = int(COOLDOWN_SECONDS / WINDOW_SIZE)

def compute_rms(signal):
    if len(signal) == 0:
        return 0
    return np.sqrt(np.mean(signal**2))

def analyze_audio():
    print("🎧 CINEMA ENGINE V6: Spike, Tail, Cooldown & Vectorized Vector Detection...")

    if not os.path.exists(AUDIO_FILE) or not os.path.exists(SEGMENTS_FILE):
        print("❌ Errore: File audio 'video.wav' o 'merged_segments.json' mancanti.")
        return False

    # Lettura e normalizzazione audio nativa
    audio, sr = sf.read(AUDIO_FILE)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)

    window_samples = int(WINDOW_SIZE * sr)
    
    # --- STEP 1: Calcolo Envelope RMS Vettorializzato (Velocità Lampo) ---
    num_chunks = len(audio) // window_samples
    if num_chunks > 0:
        truncated_audio = audio[:num_chunks * window_samples]
        reshaped = truncated_audio.reshape(num_chunks, window_samples)
        rms_array = np.sqrt(np.mean(reshaped**2, axis=1))
        rms_values = rms_array.tolist()
    else:
        rms_array = np.array([compute_rms(audio)])
        rms_values = rms_array.tolist()

    if len(rms_array) < 10:
        GLOBAL_SPIKE_THRESHOLD = np.max(rms_array) if len(rms_array) > 0 else 0.1
        GLOBAL_SILENCE_THRESHOLD = np.min(rms_array) if len(rms_array) > 0 else 0.015
    else:
        GLOBAL_SPIKE_THRESHOLD = np.percentile(rms_array, 90)
        GLOBAL_SILENCE_THRESHOLD = max(np.percentile(rms_array, 25), 0.015)

    # --- STEP 2: Carico segmenti Whisper modificati dallo Sticky Merge ---
    with open(SEGMENTS_FILE, "r", encoding="utf-8") as f:
        segments = json.load(f)

    cooldown_until_idx = -1

   # --- STEP 3: Analisi Picchi ad ampio spettro ---
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue

        start_idx = min(int(seg["start"] / WINDOW_SIZE), len(rms_values) - 1)
        end_idx = min(int(seg["end"] / WINDOW_SIZE), len(rms_values) - 1)
        
        # Inizializzazione standard obbligatoria per TUTTI i segmenti
        seg["cinematic_spike"] = False
        seg["audio_multiplier"] = 1.0

        # Se l'inizio del segmento cade dentro il cooldown audio, 
        # forziamo il salto dello spike ma manteniamo il segmento valido per il QA
        if start_idx <= cooldown_until_idx:
            continue

        # Estrazione del picco massimo in tutta la durata del blocco
        segment_windows = rms_values[start_idx:max(start_idx + 1, end_idx + 1)]
        max_rms_in_segment = np.max(segment_windows) if len(segment_windows) > 0 else rms_values[start_idx]

        # Valutazione dello spike sul picco reale della clip
        if max_rms_in_segment > GLOBAL_SPIKE_THRESHOLD and start_idx >= SILENCE_WINDOWS:
            history_window = rms_values[start_idx - SILENCE_WINDOWS:start_idx]
            
            # FIX INGEGNERISTICO: Usiamo la media (np.mean) invece del controllo assoluto (np.all)
            # Permette di rilevare gli urli anche se c'è musica di sottofondo o rumore di gioco
            if np.mean(history_window) <= GLOBAL_SILENCE_THRESHOLD * 1.5:
                seg["cinematic_spike"] = True
                spike_ratio = max_rms_in_segment / (GLOBAL_SPIKE_THRESHOLD + 1e-6)
                seg["audio_multiplier"] = float(round(min(spike_ratio, MAX_MULTIPLIER), 3))

                # Estensione della coda energetica post-urlo
                # NB: non riusare `i` (indice segmento) nel loop RMS — shadowing rompeva il clamp sul next.
                tail_end_idx = end_idx
                tail_limit_idx = min(end_idx + int(MAX_TAIL_SECONDS / WINDOW_SIZE), len(rms_values) - 1)
                seg_idx = i

                for rms_i in range(end_idx, tail_limit_idx):
                    if rms_values[rms_i] > GLOBAL_SILENCE_THRESHOLD * ENERGY_TAIL_FACTOR:
                        tail_end_idx = rms_i
                    else:
                        break

                # Aggiorniamo la fine del segmento estendendola sulla coda audio
                new_end = (tail_end_idx + 1) * WINDOW_SIZE
                old_end = float(seg["end"])
                if seg_idx + 1 < len(segments):
                    next_start = float(segments[seg_idx + 1]["start"])
                    new_end = min(new_end, next_start - 0.05)
                seg["end"] = new_end

                # Allinea words/testo alla coda audio (evita sub che finiscono prima del payoff)
                words = seg.get("words") or []
                if words:
                    last = words[-1]
                    last_end = float(last.get("end", old_end))
                    if last_end < new_end - 0.05:
                        last = dict(last)
                        last["end"] = round(new_end, 3)
                        words[-1] = last
                        seg["words"] = words

                # Il cooldown viene impostato partendo dalla NUOVA fine calcolata
                cooldown_until_idx = tail_end_idx + COOLDOWN_WINDOWS

    # --- STEP 4: Scrittura JSON UTF-8 sicuro ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print("✅ Analisi completata con successo. Generato:", OUTPUT_FILE)
    print(f"🔹 Spike threshold calcolato: {GLOBAL_SPIKE_THRESHOLD:.6f}")
    print(f"🔹 Silence baseline calcolato: {GLOBAL_SILENCE_THRESHOLD:.6f}")
    return True

if __name__ == "__main__":
    success = analyze_audio()
    if not success:
        sys.exit(1)