import os
import sys
import subprocess
import config

def extract_audio():
    input_video = config.VIDEO_FILENAME
    output_audio = os.path.join(config.OUTPUT_DIR, "video.wav")

    print(f"🔊 AUDIO ENGINE: Estrazione traccia Microfono PULITO ED ISOLATO...")

    if not os.path.exists(input_video):
        print(f"❌ Errore: Manca il video sorgente in {input_video}")
        return False

    # Controllo di integrità: salta solo se il file esiste ED è maggiore di zero byte
    if os.path.exists(output_audio) and os.path.getsize(output_audio) > 0:
        print(f"✅ Audio integro trovato ({os.path.getsize(output_audio) / 1024 / 1024:.1f} MB). Uso quello esistente.")
        return True

    cmd = [
        config.FFMPEG_BIN, "-y",
        "-i", input_video,
        "-map", f"0:a:{config.AUDIO_EXTRACT_STREAM_INDEX}",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "error",
        output_audio
    ]

    try:
        subprocess.run(cmd, check=True)
        # Controllo post-scrittura rapido
        if os.path.exists(output_audio) and os.path.getsize(output_audio) > 0:
            print(f"✅ Audio isolato estratto con successo: {output_audio}")
            return True
        else:
            print("❌ Errore: FFmpeg ha terminato ma il file prodotto è vuoto.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore FFmpeg durante l'estrazione: {e}")
        return False

if __name__ == "__main__":
    success = extract_audio()
    if not success:
        sys.exit(1)