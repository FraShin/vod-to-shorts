"""
Configurazione della pipeline: percorsi, layout video, audio, sottotitoli.

Ogni valore e' sovrascrivibile da variabile d'ambiente, e le variabili si
possono mettere in `scripts/.env` (file locale, gitignorato). Vedere
`.env.example` per la lista completa.
"""
import glob
import os
import re
import sys

# ==============================
# OUTPUT A PROVA DI WINDOWS
# ==============================
# Gli stadi stampano emoji, e main.py reindirizza l'output di ognuno in
# logs/<stadio>.py.log. Su Windows, quando stdout NON è un terminale ma un
# file, Python usa la codepage locale (cp1252) e il primo print() con un'emoji
# fa morire lo stadio con UnicodeEncodeError. Su Linux/WSL non succede, perché
# lì l'encoding di default è già UTF-8: il problema esisteva solo su Windows.
# errors="replace" fa diventare "?" un carattere non rappresentabile invece di
# interrompere la pipeline.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        # Stream non riconfigurabile (test, pipe esotiche): si prosegue.
        pass

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)


def _load_local_env():
    """
    Carica variabili da scripts/.env (solo sul tuo PC, non va su Git).
    Le variabili già impostate in Windows hanno priorità.
    """
    env_path = os.path.join(SCRIPTS_DIR, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        print(f"⚠️  {name} non è un intero valido, uso {default}.")
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        print(f"⚠️  {name} non è un numero valido, uso {default}.")
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


# ==============================
# PERCORSI
# ==============================
# Struttura attesa del progetto:
#   <root>/input/          VOD sorgente .mp4
#   <root>/output/         JSON intermedi + video.wav
#   <root>/clips/          clip prodotte
#   <root>/clips_archive/  clip delle run precedenti
#   <root>/logs/           log di ogni stadio
INPUT_DIR = os.environ.get("INPUT_DIR") or os.path.join(_PROJECT_ROOT, "input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(_PROJECT_ROOT, "output")
CLIPS_OUTPUT_FOLDER = os.environ.get("CLIPS_DIR") or os.path.join(_PROJECT_ROOT, "clips")
CLIPS_ARCHIVE_DIR = os.environ.get("CLIPS_ARCHIVE_DIR") or os.path.join(_PROJECT_ROOT, "clips_archive")
LOGS_DIR = os.environ.get("LOGS_DIR") or os.path.join(_PROJECT_ROOT, "logs")

# ==============================
# LAYOUT VERTICALE (1080x1920)
# ==============================
# Ritaglio della faccia e del gioco dal VOD orizzontale. I valori sotto sono
# tarati su una registrazione OBS specifica: se il tuo video è composto
# diversamente, cambiali (o mettili nel .env) guardando un frame del VOD.
#
#   SOURCE_*  rettangolo della webcam nel video sorgente
#   TARGET_*  altezza finale delle due fasce: cam sopra, gioco sotto
#             (dovrebbe valere TARGET_CAM_H + TARGET_GAME_H = 1920)
SOURCE_X = _env_int("SOURCE_X", 1)
SOURCE_Y = _env_int("SOURCE_Y", 490)
SOURCE_W = _env_int("SOURCE_W", 409)
SOURCE_H = _env_int("SOURCE_H", 348)
TARGET_CAM_H = _env_int("TARGET_CAM_H", 768)
TARGET_GAME_H = _env_int("TARGET_GAME_H", 1152)

# Output: fps e qualità costante. Il game viene ritagliato in 16:9 partendo
# dalla larghezza piena del video, poi scalato a 1080.
VIDEO_FPS = _env_int("VIDEO_FPS", 60)
VIDEO_CQ = _env_int("VIDEO_CQ", 22)

# Binario ffmpeg da usare. Se è nel PATH va benissimo il default; su Windows
# spesso non c'è, e allora qui puoi indicare il percorso completo, es:
#   FFMPEG_BIN=C:/ffmpeg/bin/ffmpeg.exe
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

# ==============================
# TRACCE AUDIO DEL MP4 (0 = prima traccia)
# ==============================
# OBS registra più tracce separate: tipicamente mic e audio di gioco.
# Se il mix esce sbagliato, guarda quali tracce contiene il VOD con:
#   ffprobe -v error -select_streams a -show_entries stream=index,codec_name <vod.mp4>
AUDIO_MIC_STREAM_INDEX = _env_int("AUDIO_MIC_STREAM_INDEX", 1)
AUDIO_GAME_STREAM_INDEX = _env_int("AUDIO_GAME_STREAM_INDEX", 2)
AUDIO_EXTRACT_STREAM_INDEX = _env_int("AUDIO_EXTRACT_STREAM_INDEX", 1)
# Mix Short: voce in avanti, game udibile (non “sottofondo lontano”).
# Override in scripts/.env — es. GAME_VOLUME=1.0 se vuoi più vicino al VOD OBS.
MIC_VOLUME = _env_float("MIC_VOLUME", 1.2)
GAME_VOLUME = _env_float("GAME_VOLUME", 0.85)

# ==============================
# QUANTE CLIP PRODURRE
# ==============================
TARGET_TOTAL_CLIPS = _env_int("TARGET_TOTAL_CLIPS", 10)

# ==============================
# MARKER DEL PICCO ENERGETICO (segnale interno)
# ==============================
# Quando il rilevatore acustico trova un picco, transcribe_engine appende questo
# marker al testo del segmento. Il marker NON è una frase dello streamer: è un
# segnale che il codice si scrive da solo e rilegge negli stadi successivi
# (merge, scoring, judge, sottotitoli). Per questo sta qui e non nel vocabulary.
#
#   MARKER → come appare dentro il testo del segmento
#   TOKEN  → forma normalizzata, per la ricerca sul testo normalizzato
#   LABEL  → etichetta leggibile, sostituisce il marker nel testo finale
ENERGY_SPIKE_MARKER = os.environ.get("ENERGY_SPIKE_MARKER", "[ENERGY_SPIKE]")
ENERGY_SPIKE_TOKEN = ENERGY_SPIKE_MARKER.strip("[]").strip().lower().replace(" ", "_")
ENERGY_SPIKE_LABEL = os.environ.get("ENERGY_SPIKE_LABEL", "ENERGY SPIKE!")
# Bonus di punteggio per un segmento con picco: serve a fargli superare la
# soglia dinamica anche se è corto o ha poche parole.
ENERGY_SPIKE_SCORE_BONUS = _env_float("ENERGY_SPIKE_SCORE_BONUS", 15.0)

# ==============================
# ANTIFRAGMENTAZIONE DEGLI SPIKE
# ==============================
# Due stadi usano lo stesso valore: audio_analysis lo applica, qa_validator
# disfa gli spike abusivi. Se i due valori divergono, il QA smonta spike
# legittimi: per questo la costante è una sola.
COOLDOWN_SECONDS = _env_float("COOLDOWN_SECONDS", 4.0)

# ==============================
# FFMPEG: TIMEOUT
# ==============================
# Il timeout cresce con la durata della clip: un taglio da 60s su un VOD lungo
# può richiedere parecchio, mentre un timeout fisso taglierebbe via le clip
# grosse a caso.
FFMPEG_TIMEOUT_BASE = _env_int("FFMPEG_TIMEOUT_BASE", 600)
FFMPEG_TIMEOUT_PER_SEC = _env_int("FFMPEG_TIMEOUT_PER_SEC", 12)

# ==============================
# CODEC VIDEO
# ==============================
# Codifica hardware NVIDIA. Con USE_NVENC=0 la pipeline usa libx264 su CPU:
# molto più lenta, ma funziona anche su AMD/Intel/Mac e su macchine senza
# driver CUDA. Il filtro verticale resta identico.
USE_NVENC = _env_bool("USE_NVENC", True)

# ==============================
# CREDENZIALI (env ha priorità)
# ==============================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
# Chiave: variabile d'ambiente DEEPSEEK_API_KEY oppure stringa nel secondo argomento sotto
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
# Modello API (docs 2026): deepseek-v4-flash | deepseek-v4-pro
# deepseek-chat funziona ancora fino ~24/07/2026 poi punta a v4-flash
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

# ==============================
# TRASCRIZIONE
# ==============================
# Lingua parlata nel VOD. Non metterla anche nel vocabulary: è configurazione,
# non gusto, e averla in due posti significa averne due fonti di verità.
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "it").strip().lower()
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3").strip()

# ==============================
# SOTTOTITOLI
# ==============================
# Ritardo globale sottotitoli (secondi). Tieni basso: 0.08–0.14. NON usare 0.5+.
SUB_SYNC_LAG_SEC = _env_float("SUB_SYNC_LAG_SEC", 0.10)
# clip = Whisper su ogni clip (consigliato per burn-in) | vod = transcription.json (manual_cutter)
SUB_WORD_SOURCE = os.environ.get("SUB_WORD_SOURCE", "clip").strip().lower()
SUB_CLIP_WORD_TIMING = os.environ.get("SUB_CLIP_WORD_TIMING", "whisper").strip().lower()
# Chunk: max parole a schermo se gap < SUB_CHUNK_GAP_SEC
SUB_CHUNK_MAX_WORDS = _env_int("SUB_CHUNK_MAX_WORDS", 3)
SUB_CHUNK_GAP_SEC = _env_float("SUB_CHUNK_GAP_SEC", 0.35)
# tiktok = attiva gialla + altre grigie | all_yellow = 2-3 parole tutte gialle
SUB_MULTI_WORD_STYLE = os.environ.get("SUB_MULTI_WORD_STYLE", "tiktok").strip().lower()
# Larghezza riga 1080 @ font 95: non superare ~22 caratteri (spazi inclusi) nel chunk
SUB_CHUNK_MAX_CHARS = _env_int("SUB_CHUNK_MAX_CHARS", 22)
# Parola più lunga di così non si affianca ad altre nello stesso chunk
SUB_CHUNK_LONG_WORD_CHARS = _env_int("SUB_CHUNK_LONG_WORD_CHARS", 11)


def get_latest_vod():
    """Rileva dinamicamente l'ultimo MP4 inserito senza bloccare l'importazione.

    Se la variabile d'ambiente VOD_FILE è impostata (main.py la imposta quando
    usi --vod) quella ha la precedenza: ogni stadio gira in un processo separato
    e deve poter sapere quale VOD sta lavorando.
    """
    forzato = os.environ.get("VOD_FILE")
    if forzato:
        return forzato

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        return ""

    vods = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    if not vods:
        # Ritorna stringa vuota invece di crashare brutalmente l'import dei moduli
        return ""

    latest = max(vods, key=os.path.getmtime)
    return latest


def refresh_vod():
    """Aggiorna il VOD attivo (chiamare all'inizio pipeline)."""
    global VIDEO_FILENAME
    VIDEO_FILENAME = get_latest_vod()
    return VIDEO_FILENAME


def ffmpeg_vertical_stack_filter():
    """Filtro video: webcam sopra, gioco sotto, formato verticale 1080x1920."""
    return (
        f"[0:v]setpts=PTS-STARTPTS,crop={SOURCE_W}:{SOURCE_H}:{SOURCE_X}:{SOURCE_Y},"
        f"scale=1080:-1,crop=1080:{TARGET_CAM_H}:0:(ih-{TARGET_CAM_H})/2[cam];"
        f"[0:v]setpts=PTS-STARTPTS,crop=ih*0.9375:ih:(iw-ow)/2:0,"
        f"scale=1080:{TARGET_GAME_H}[game];"
        f"[cam][game]vstack,fps=fps={VIDEO_FPS}[v_stack]"
    )


def ffmpeg_mix_audio_filter():
    mi = AUDIO_MIC_STREAM_INDEX
    gi = AUDIO_GAME_STREAM_INDEX
    # VOD con una sola traccia audio (microfono e gioco già mixati a monte, come
    # nei file ritagliati o ri-esportati): indicizzare due volte la stessa traccia
    # e passarla ad amix sommerebbe il segnale con sé stesso, raddoppiandolo e
    # saturando. In quel caso serve un solo volume, non un mix.
    if mi == gi:
        return f"[0:a:{mi}]asetpts=PTS-STARTPTS,volume={MIC_VOLUME}[a]"
    # normalize=0: i coefficienti volume= sono quelli effettivi (senza dimezzamento amix).
    return (
        f"[0:a:{mi}]asetpts=PTS-STARTPTS,volume={MIC_VOLUME}[mic];"
        f"[0:a:{gi}]asetpts=PTS-STARTPTS,volume={GAME_VOLUME}[game_audio];"
        f"[mic][game_audio]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[a]"
    )


def ffmpeg_input_args(use_gpu: bool | None = None):
    """Argomenti di ingresso: decodifica hardware solo se useremo NVENC."""
    if use_gpu is None:
        use_gpu = USE_NVENC
    return ["-hwaccel", "cuda"] if use_gpu else []


def ffmpeg_video_encoder_args():
    """Codifica video: NVENC (NVIDIA, veloce) o libx264 (CPU, universale)."""
    if USE_NVENC:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(VIDEO_CQ)]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(VIDEO_CQ)]


def ffmpeg_timeout(duration_sec: float) -> int:
    return int(FFMPEG_TIMEOUT_BASE + max(0.0, duration_sec) * FFMPEG_TIMEOUT_PER_SEC)


def ffmpeg_subtitle_path(ass_path: str) -> str:
    """Prepara il percorso di un .ass per il filtro `subtitles=` di ffmpeg.

    Due cose vanno sistemate, e la seconda è controintuitiva:

    1. Su Windows il percorso contiene ':' (C:/...), che per il parser dei
       filtri di ffmpeg è il separatore fra opzioni.
    2. L'escape vuole DUE backslash, non uno. Il primo livello di escape lo
       consuma il parser del filtergraph, il secondo il parser del filtro.
       Con un solo backslash ffmpeg legge "C" come nome del file e tutto il
       resto come valore di `original_size`, fallendo con:
           Unable to parse "original_size" option value "..." as image size
       Le due varianti sono state provate su ffmpeg 9.0.1: un backslash fallisce,
       due funzionano.

    Su Linux/WSL il percorso non contiene ':' e questa funzione non cambia
    nulla: è il motivo per cui il bug si manifestava solo su Windows.
    """
    p = os.path.abspath(ass_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = p.replace(":", "\\\\:", 1)
    return p


def clean_filename(text):
    text = text.replace("è", "e").replace("é", "e").replace("à", "a")
    text = text.replace("ò", "o").replace("ù", "u").replace("ì", "i")
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    clean = clean.replace(" ", "_").strip("_")
    return clean[:30] if clean else "clip"


# Script eseguiti singolarmente: risolvi subito il VOD
VIDEO_FILENAME = get_latest_vod()
