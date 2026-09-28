import os
import json
import time
import sys
import wave
import numpy as np
import torch
from faster_whisper import WhisperModel

import config
import vocabulary

# ==============================
# LIBRERIE CUDA: dove trovarle
# ==============================
# CTranslate2 (il motore di faster-whisper) non cerca le librerie CUDA dentro i
# pacchetti pip `nvidia-*`: vanno rese visibili esplicitamente, e il modo giusto
# dipende dal sistema operativo.
#
#   Linux/WSL -> i .so finiscono in site-packages/nvidia/<lib>/lib e si
#                annunciano con LD_LIBRARY_PATH.
#   Windows   -> i .dll finiscono in site-packages/nvidia/<lib>/bin, e dal
#                Python 3.8 il PATH non basta più: serve os.add_dll_directory,
#                il cui handle va tenuto vivo o il garbage collector chiude la
#                cartella e le DLL tornano invisibili.
#
# Se non trova niente (CUDA di sistema, Mac, o solo CPU) non fa nulla: la
# pipeline degrada su CPU invece di non partire.
_DLL_HANDLES = []


def _expose_cuda_libraries():
    import glob
    import sysconfig

    candidati = [os.path.join(os.path.dirname(torch.__file__), "lib")]
    for radice in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
        candidati += glob.glob(os.path.join(radice, "nvidia", "*", "bin"))
        candidati += glob.glob(os.path.join(radice, "nvidia", "*", "lib"))
    candidati = [c for c in candidati if os.path.isdir(c)]
    if not candidati:
        return

    if os.name == "nt":
        for cartella in candidati:
            try:
                _DLL_HANDLES.append(os.add_dll_directory(cartella))
            except (AttributeError, OSError):
                pass
    else:
        esistente = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            ([esistente] if esistente else []) + candidati
        )


_expose_cuda_libraries()


def detect_audio_energy_spikes(wav_path, threshold_db=-12.0, window_ms=500) -> list:
    """
    FIX INGEGNERISTICO 2: Analisi RMS (Root Mean Square) nativa dell'audio.
    Intercetta i picchi di decibel (urla, colpi di scena) dove Whisper va in clipping.
    """
    if not os.path.exists(wav_path):
        print(f"⚠️ Spike Detector: File {wav_path} non trovato per l'analisi RMS.")
        return []
    
    print("🔊 Avvio Audio Energy Spike Detector (RMS Analisi)...")
    try:
        with wave.open(wav_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            
            # Lettura dei frame binari
            raw_bytes = wf.readframes(num_frames)
            
            # Mappatura dinamica del tipo di dato PCM
            if sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            else:
                dtype = np.uint8
                
            data = np.frombuffer(raw_bytes, dtype=dtype)
            
            # Se stereo, facciamo il downmix a mono prendendo la media dei canali
            if channels > 1:
                data = data.reshape(-1, channels).mean(axis=1)
            
            # Normalizzazione dello spettrogramma tra -1.0 e 1.0
            max_val = np.max(np.abs(data))
            if max_val == 0:
                return []
            normalized_data = data / max_val
            
            # Calcolo della finestra in campioni discreti
            window_samples = int(sample_rate * (window_ms / 1000.0))
            spike_timestamps = []
            
            for i in range(0, len(normalized_data), window_samples):
                chunk = normalized_data[i:i+window_samples]
                if len(chunk) == 0:
                    continue
                
                # Calcolo matematico dell'energia RMS
                rms = np.sqrt(np.mean(chunk**2))
                # Conversione in Decibel Full Scale (dBFS)
                db = 20 * np.log10(rms) if rms > 0 else -100
                
                # Registrazione del timestamp se supera la tolleranza di clipping
                if db > threshold_db:
                    timestamp_sec = round(i / sample_rate, 3)
                    spike_timestamps.append(timestamp_sec)
                    
            print(f"🎯 Spike Detector completato: Rilevati {len(spike_timestamps)} picchi di energia acustica.")
            return spike_timestamps
            
    except Exception as e:
        print(f"⚠️ Errore non critico nell'estrazione dei picchi RMS: {e}")
        return []


def transcribe_audio():
    # Punta al file standardizzato video.wav
    input_audio = os.path.join(config.OUTPUT_DIR, "video.wav")
    output_file = os.path.join(config.OUTPUT_DIR, "transcription.json")

    print("🎙️  FASTER-WHISPER TRANSCRIPTION ENGINE (V3 - ANTI-BLEEDING ACTIVE)")
    print("==================================================")

    # 1. Rilevamento hardware dinamico
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU Rilevata: {gpu_name} — VRAM allocata per {config.WHISPER_MODEL}")
    else:
        device = "cpu"
        compute_type = "int8"
        print("⚠️ ATTENZIONE: GPU non rilevata o driver CUDA non pronti. Fallback su CPU Mode.")

    if not os.path.exists(input_audio) or os.path.getsize(input_audio) == 0:
        print(f"❌ File audio mancante o corrotto: {input_audio}")
        return False

    # Eseguiamo lo Spike Detector prima dell'inferenza del modello
    detected_spikes = detect_audio_energy_spikes(input_audio, threshold_db=-11.0, window_ms=500)

    try:
        print(f"⏳ Caricamento modello '{config.WHISPER_MODEL}' in memoria GPU...")
        model = WhisperModel(
            config.WHISPER_MODEL, device=device, compute_type=compute_type, cpu_threads=4
        )
    except Exception as e:
        print(f"❌ Errore critico nel caricamento del modello Whisper: {e}")
        return False

    print(f"⏳ Inizio trascrizione ({config.WHISPER_LANGUAGE} | Word Timestamps Attivi)...")
    start_time = time.time()

    try:
        # Il prompt di contesto (nomi propri, gergo, tormentoni) arriva dal
        # vocabulary: è specifico di chi streamma, il codice resta generico.
        prompt_vod = vocabulary.whisper_prompt("vod")
        extra_kwargs = {"initial_prompt": prompt_vod} if prompt_vod else {}

        # 🔥 PARAMETRI VAD COERENTI CON LA SELEZIONE AGENT PRECEDENTE
        segments_iter, info = model.transcribe(
            input_audio,
            language=config.WHISPER_LANGUAGE,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,                  # Sensibilità standard di attivazione
                min_silence_duration_ms=250,    # Isola i micro-silenzi dello stream
                speech_pad_ms=50                # Abbassato per limitare il bleeding
            ),
            **extra_kwargs
        )

        print(f"🌍 Lingua verificata: {info.language} (Accuratezza: {info.language_probability:.2f})")
        print("💾 Trascrizione e fusione flussi acustici in corso...")

        segments_list = []
        for seg in segments_iter:
            words_list = []
            seg_start = round(seg.start, 3)
            seg_end = round(seg.end, 3)
            seg_text = seg.text.strip()

            # Estrazione delle parole reali trascritte da Whisper
            if getattr(seg, 'words', None):
                for w in seg.words:
                    w_start = round(w.start, 3)
                    w_end = round(w.end, 3)

                    # 🎯 SAFETY HARD-CLAMP
                    w_start = max(seg_start, w_start)
                    w_end = min(seg_end, w_end)

                    words_list.append({
                        "start": w_start,
                        "end": w_end,
                        "word": w.word,
                        "probability": round(w.probability, 3)
                    })

            # Picchi RMS: solo metadata + marker nel testo per scoring.
            # NON iniettare parole fake in words[] (sporcano ASS se SUB_WORD_SOURCE=vod).
            energy_spikes = []
            for spike_time in detected_spikes:
                if seg_start <= spike_time <= seg_end:
                    if not any(abs(s - spike_time) < 1.0 for s in energy_spikes):
                        energy_spikes.append(round(spike_time, 3))

            if energy_spikes:
                seg_text += f" {config.ENERGY_SPIKE_MARKER}"

            segments_list.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "words": words_list,
                "energy_spikes": energy_spikes,
            })
            print(f"   [{seg_start:.1f}s → {seg_end:.1f}s] {seg_text}")

        result = {
            "source_vod": os.path.basename(config.VIDEO_FILENAME),
            "source_vod_path": config.VIDEO_FILENAME,
            "segments": segments_list,
        }

        print("💾 Scrittura del file di trascrizione modificato...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        duration = time.time() - start_time
        print(f"✅ Trascrizione e iniezione RMS completate in {duration:.2f} secondi.")
        print(f"📝 Totale segmenti elaborati: {len(segments_list)}")
        return True

    except Exception as e:
        print(f"❌ Errore critico durante l'elaborazione dei segmenti: {e}")
        return False


if __name__ == "__main__":
    success = transcribe_audio()
    if not success:
        sys.exit(1)


# --- Trascrizione per singola clip (sottotitoli allineati all'audio del file tagliato) ---

_whisper_model: WhisperModel | None = None


def get_shared_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
        print(f"   🎙️  Clip Whisper: GPU {config.WHISPER_MODEL} (caricamento una volta per run cut_engine)...")
    else:
        device, compute_type = "cpu", "int8"
        print(f"   🎙️  Clip Whisper: CPU {config.WHISPER_MODEL}...")
    _whisper_model = WhisperModel(
        config.WHISPER_MODEL, device=device, compute_type=compute_type, cpu_threads=4
    )
    return _whisper_model


def transcribe_clip_wav(wav_path: str) -> list[dict]:
    """Ritorna segmenti con words (timestamp 0 = inizio clip)."""
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
        return []

    model = get_shared_whisper_model()
    # Stesso principio del VOD: il contesto è dato dal vocabulary, non dal codice.
    prompt_clip = vocabulary.whisper_prompt("clip")
    extra_kwargs = {"initial_prompt": prompt_clip} if prompt_clip else {}

    segments_iter, _info = model.transcribe(
        wav_path,
        language=config.WHISPER_LANGUAGE,
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
        **extra_kwargs
    )

    segments_list: list[dict] = []
    for seg in segments_iter:
        words_list = []
        seg_start = round(seg.start, 3)
        seg_end = round(seg.end, 3)
        if getattr(seg, "words", None):
            for w in seg.words:
                w_start = max(seg_start, round(w.start, 3))
                w_end = min(seg_end, round(w.end, 3))
                if w_end <= w_start:
                    continue
                words_list.append({
                    "start": w_start,
                    "end": w_end,
                    "word": w.word,
                    "probability": round(w.probability, 3),
                })
        if words_list:
            segments_list.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg.text.strip(),
                "words": words_list,
            })
    return segments_list