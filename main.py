"""
Orchestratore della pipeline: dal VOD alle clip verticali con sottotitoli.

Ogni stadio gira in un processo separato (vedi PIPELINE più sotto) e comunica
con gli altri attraverso file JSON nella cartella output/. Questo file decide
l'ordine, salta gli stadi già completati e si ferma al primo errore vero.

Uso:
    python main.py                     # ultimo .mp4 in input/, 10 clip
    python main.py --vod vod.mp4       # VOD specifico
    python main.py --clips 5           # quante clip
    python main.py --no-wipe           # non svuotare output/ e clips/
    python main.py --check             # verifica l'ambiente ed esci
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import config
import vocabulary

# ==============================
# PERCORSI E CONFIGURAZIONI
# ==============================
SCRIPTS_DIR = config.SCRIPTS_DIR
OUTPUT_DIR = config.OUTPUT_DIR
CLIPS_DIR = config.CLIPS_OUTPUT_FOLDER
LOGS_DIR = config.LOGS_DIR

def script_path(name):
    return os.path.join(SCRIPTS_DIR, name)

def output_path(name):
    return os.path.join(OUTPUT_DIR, name) if name else ""


def wipe_work_dirs():
    """Svuota output/ e clips/ all'avvio (disabilita con PIPELINE_NO_WIPE=1)."""
    for folder in (OUTPUT_DIR, CLIPS_DIR):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    print("🧹 Cartelle output e clips azzerate per nuova run.\n")


# ==============================
# RIGA DI COMANDO E VERIFICA AMBIENTE
# ==============================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Pipeline VOD -> Shorts: trasforma un VOD in clip verticali sottotitolate.",
        epilog="Senza argomenti usa l'ultimo .mp4 presente in input/ e produce "
               f"{config.TARGET_TOTAL_CLIPS} clip.",
    )
    parser.add_argument(
        "--vod", metavar="FILE",
        help="VOD da processare. Default: l'ultimo .mp4 in input/",
    )
    parser.add_argument(
        "--clips", type=int, metavar="N",
        help=f"quante clip produrre. Default: {config.TARGET_TOTAL_CLIPS}",
    )
    parser.add_argument(
        "--no-wipe", action="store_true",
        help="non svuotare output/ e clips/ all'avvio (utile per riprendere un run)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="verifica l'ambiente (ffmpeg, CUDA, encoder, VOD, vocabulary) ed esce",
    )
    return parser.parse_args(argv)


def check_environment():
    """Controlla l'ambiente prima di lanciare la pipeline. Non modifica niente.

    Serve a scoprire adesso, in due secondi, quello che altrimenti si scopre
    venti minuti dopo: ffmpeg senza NVENC, CUDA assente, libass mancante,
    nessun VOD in input/.
    """
    print("\n" + "=" * 60)
    print("🔎 VERIFICA AMBIENTE")
    print("=" * 60)

    problemi: list[str] = []
    avvisi: list[str] = []

    def esito(ok, etichetta, dettaglio="", bloccante=True):
        icona = "✅" if ok else ("❌" if bloccante else "⚠️ ")
        suffisso = f" — {dettaglio}" if dettaglio else ""
        print(f"  {icona} {etichetta}{suffisso}")
        if not ok:
            (problemi if bloccante else avvisi).append(etichetta)

    v = sys.version_info
    esito(v >= (3, 11), f"Python {v.major}.{v.minor}.{v.micro}", "serve 3.11 o superiore")

    # FFMPEG_BIN può essere il nome ("ffmpeg", cercato nel PATH) oppure un
    # percorso completo: su Windows il PATH non lo contiene quasi mai.
    ffmpeg = shutil.which(config.FFMPEG_BIN) or (
        config.FFMPEG_BIN if os.path.isfile(config.FFMPEG_BIN) else None
    )
    esito(bool(ffmpeg), f"ffmpeg ({config.FFMPEG_BIN})",
          "" if ffmpeg else "non trovato: installalo o imposta FFMPEG_BIN in .env")

    # ffprobe sta sempre accanto a ffmpeg: se ffmpeg è assoluto e ffprobe non è
    # nel PATH, lo cerchiamo nella stessa cartella.
    ffprobe = shutil.which("ffprobe")
    if not ffprobe and ffmpeg:
        accanto = os.path.join(
            os.path.dirname(ffmpeg), "ffprobe.exe" if os.name == "nt" else "ffprobe"
        )
        ffprobe = accanto if os.path.isfile(accanto) else None
    esito(bool(ffprobe), "ffprobe",
          "" if ffprobe else "non trovato: serve per leggere le tracce audio")

    if ffmpeg:
        try:
            encoder = subprocess.run(
                [config.FFMPEG_BIN, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=30,
            ).stdout
            if config.USE_NVENC:
                nvenc_ok = "h264_nvenc" in encoder
                esito(nvenc_ok, "encoder h264_nvenc",
                      "" if nvenc_ok else "imposta USE_NVENC=0 per usare libx264 su CPU",
                      bloccante=False)
            else:
                esito("libx264" in encoder, "encoder libx264")

            filtri = subprocess.run(
                [config.FFMPEG_BIN, "-hide_banner", "-filters"],
                capture_output=True, text=True, timeout=30,
            ).stdout
            libass_ok = "subtitles" in filtri
            esito(libass_ok, "filtro subtitles (libass)",
                  "" if libass_ok else "senza libass il burn-in dei sottotitoli non funziona")
        except (OSError, subprocess.SubprocessError) as e:
            esito(False, "interrogazione di ffmpeg", str(e))

    try:
        import torch
        esito(True, f"torch {torch.__version__}")
        if torch.cuda.is_available():
            esito(True, "CUDA disponibile", torch.cuda.get_device_name(0))
        else:
            esito(False, "CUDA disponibile",
                  "la trascrizione userà la CPU: funziona ma è molto più lenta",
                  bloccante=False)
    except ImportError:
        esito(False, "torch installato", "pip install -r requirements.txt")

    try:
        import faster_whisper  # noqa: F401
        esito(True, "faster-whisper installato")
    except ImportError:
        esito(False, "faster-whisper installato", "pip install -r requirements.txt")

    somma = config.TARGET_CAM_H + config.TARGET_GAME_H
    layout_ok = somma == 1920
    esito(layout_ok, f"layout verticale {config.TARGET_CAM_H} + {config.TARGET_GAME_H} = {somma}",
          "" if layout_ok else "dovrebbe fare 1920 (vedi SOURCE_*/TARGET_* in .env)",
          bloccante=False)

    config.refresh_vod()
    if config.VIDEO_FILENAME:
        esito(True, "VOD trovato", os.path.basename(config.VIDEO_FILENAME))
    else:
        esito(False, "VOD trovato",
              f"nessun .mp4 in {config.INPUT_DIR} (mettine uno lì oppure usa --vod)")

    if os.path.isfile(vocabulary.VOCABULARY_FILE):
        dati = vocabulary.vocabulary()
        voci = (sum(len(x) for x in dati["triggers"].values())
                + len(dati["hype"]["words"])
                + len(dati["hype"]["phrases"])
                + len(dati["slang_fixes"]))
        esito(voci > 0, "vocabulary.json caricato", f"{voci} voci di gusto",
              bloccante=False)
    else:
        esito(False, "vocabulary.json presente",
              "assente: nessuna catch-phrase riconosciuta (copia vocabulary.example.json)",
              bloccante=False)

    api_ok = bool(config.DEEPSEEK_API_KEY)
    esito(api_ok, "DEEPSEEK_API_KEY configurata",
          "" if api_ok else "senza chiave l'agente ripiega su Spike Recovery + Hype Detector",
          bloccante=False)

    tg_ok = bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)
    esito(tg_ok, "Telegram configurato",
          "" if tg_ok else "nessuna notifica a fine run", bloccante=False)

    print()
    if problemi:
        print(f"❌ {len(problemi)} problemi bloccanti: {', '.join(problemi)}")
        print("   Da sistemare prima di lanciare la pipeline.")
        return 1
    if avvisi:
        print(f"⚠️  {len(avvisi)} avvisi non bloccanti: {', '.join(avvisi)}")
        print("   La pipeline può girare lo stesso.")
        return 0
    print("✅ Ambiente pronto.")
    return 0


# ==============================
# GESTIONE HARDWARE (GPU STABILIZATION)
# ==============================
def attesa_stabilizzazione_gpu():
    """
    Sostituisce il vecchio libera_vram. Poiché subprocess libera già la VRAM,
    questa pausa serve unicamente a dare al driver NVIDIA il tempo di riposare
    tra un calcolo massivo e l'altro in WSL2.
    """
    print("⏳ Pausa di stabilizzazione per il driver GPU (2s)...")
    time.sleep(2)
    return True

# ==============================
# PIPELINE CLOUD-NATIVE
# ==============================
PIPELINE = [
    ("audio_engine.py",      "Estrazione Audio",             "video.wav"),
    ("transcribe_engine.py", "Trascrizione Whisper",         "transcription.json"),
    (attesa_stabilizzazione_gpu, "GPU Cooldown Check",       ""),
    ("merge_segments.py",    "Merge Testuale",               "merged_segments.json"),
    ("audio_analysis.py",    "Cinema Engine",                "segments_with_audio.json"),
    ("score_segments.py",    "Scoring NLP",                  "scored_segments.json"),
    ("qa_validator.py",      "QA Validation",                ""),
    ("judge_agent.py",       "Judge Agent AI (DeepSeek)",    "final_choice.json"),
    ("cut_engine.py",        "Produzione Video",             ""),
]

# ==============================
# MOTORE DI ESECUZIONE
# ==============================
def is_valid_json(file_path):
    """Verifica se il file generato è un JSON integro e non tronco"""
    if not file_path.endswith('.json'):
        return True # Se non è JSON (es. video.wav), ci fidiamo dell'esistenza
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False

def esegui_script(script, description, output_file=""):
    if callable(script):
        print(f"⚙️  {description}...")
        script()
        print(f"✅ {description} completato.\n")
        return True

    full_script = script_path(script)
    full_output = output_path(output_file) if output_file else ""

    # Smart Skip con controllo integrità
    if full_output and os.path.exists(full_output):
        if os.path.getsize(full_output) > 0 and is_valid_json(full_output):
            print(f"⏩ {description} → SALTATO (file esistente e valido).")
            return True

    print(f"⏳ {description}...")
    start_time = time.time()
    log_file = os.path.join(LOGS_DIR, f"{script}.log")

    try:
        # encoding esplicito perché il log contiene emoji.
        # PYTHONUTF8 / PYTHONIOENCODING: ogni stadio è un processo separato, e su
        # Windows con l'output reindirizzato su file Python userebbe cp1252,
        # facendo morire il primo print() con un'emoji (UnicodeEncodeError).
        with open(log_file, "w", encoding="utf-8", errors="replace") as log:
            result = subprocess.run(
                [sys.executable, full_script],
                timeout=7200,
                stdout=log,
                stderr=log,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Completato in {duration:.1f}s.\n")
            return True
        elif full_output and os.path.exists(full_output) and is_valid_json(full_output):
            # Tolleranza d'uscita solo se il file prodotto è strutturalmente perfetto
            print(f"⚠️  Crash di chiusura ignorato — File JSON integro. Proseguo.\n")
            return True
        else:
            print(f"❌ ERRORE CRITICO in {script} → codice {result.returncode}")
            print(f"   📋 Controlla i log in: {log_file}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT: {script} ha superato il limite massimo.")
        return False
    except Exception as e:
        print(f"❌ ERRORE DI SISTEMA: {e}")
        return False

# ==============================
# NOTIFICA HERMES (TELEGRAM)
# ==============================
def notifica_completamento(successo=True):
    clips = glob.glob(os.path.join(CLIPS_DIR, "*.mp4"))
    n = len(clips)

    if not successo:
        msg = f"⚠️ Pipeline interrotta. Controlla i log in {LOGS_DIR}"
    elif n == 0:
        msg = "⚠️ Pipeline completata ma nessuna clip trovata!"
    else:
        msg = f"Pipeline completata! 🎉 {n} clip pronte in {CLIPS_DIR}"

    print(f"HERMES_NOTIFY: {msg}")

    try:
        bot_token = getattr(config, "TELEGRAM_BOT_TOKEN", None)
        chat_id = getattr(config, "TELEGRAM_CHAT_ID", None)

        if bot_token and chat_id:
            # Import pigro: `requests` serve solo per Telegram. Se non è
            # installata, la pipeline non deve morire per una notifica.
            try:
                import requests
            except ImportError:
                print("⚠️  Telegram: libreria 'requests' assente (pip install requests).")
                return

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {"chat_id": chat_id, "text": msg}

            for i in range(3):
                try:
                    requests.post(url, json=payload, timeout=5)
                    print("🚀 Telegram inviato")
                    break
                except Exception as e:
                    print(f"Tentativo {i+1} fallito:", e)
                    time.sleep(1)
        else:
            print("⚠️ Token Telegram mancanti nel file config")

    except Exception as e:
        print(f"❌ Errore Telegram: {e}")

# ==============================
# MAIN POINT
# ==============================
def main(args=None):
    if args is None:
        args = parse_args()

    if args.check:
        sys.exit(check_environment())

    for folder in [OUTPUT_DIR, CLIPS_DIR, LOGS_DIR]:
        os.makedirs(folder, exist_ok=True)

    # Le variabili d'ambiente vengono ereditate dai sottoprocessi: è così che
    # --vod e --clips raggiungono ogni stadio senza passarli a mano.
    if args.vod:
        if not os.path.exists(args.vod):
            print(f"❌ VOD indicato non trovato: {args.vod}")
            sys.exit(1)
        os.environ["VOD_FILE"] = os.path.abspath(args.vod)

    if args.clips is not None:
        if args.clips < 1:
            print("❌ --clips deve essere almeno 1.")
            sys.exit(1)
        os.environ["TARGET_TOTAL_CLIPS"] = str(args.clips)
        config.TARGET_TOTAL_CLIPS = args.clips

    # Il VOD si risolve PRIMA di svuotare: non ha senso buttare la run
    # precedente per poi scoprire che non c'è niente da processare.
    config.refresh_vod()
    if not config.VIDEO_FILENAME or not os.path.exists(config.VIDEO_FILENAME):
        print(f"❌ Nessun VOD .mp4 trovato in {config.INPUT_DIR}/")
        print("   Metti un file in input/ oppure indica il percorso con --vod FILE.")
        sys.exit(1)

    no_wipe = args.no_wipe or os.environ.get("PIPELINE_NO_WIPE", "").lower() in ("1", "true", "yes")
    if not no_wipe:
        wipe_work_dirs()

    print("\n" + "=" * 50)
    print("🚀 AI STREAM MANAGER — VOD → Shorts Pipeline")
    print(f"🎬 VOD: {os.path.basename(config.VIDEO_FILENAME)}")
    print(f"🎯 Clip da produrre: {config.TARGET_TOTAL_CLIPS}")
    print("=" * 50 + "\n")

    for script, desc, outfile in PIPELINE:
        print("-" * 40)
        success = esegui_script(script, desc, outfile)

        if not success:
            print("\n" + "!"*50)
            print(f"⛔ BLOCCATO su: {script}")
            print("!"*50)
            notifica_completamento(successo=False)
            sys.exit(1) # Ritorna un exit code d'errore all'OS o a Hermes

    print("\n" + "="*50)
    print("🎉 PIPELINE COMPLETATA CON SUCCESSO!")
    print("="*50)
    notifica_completamento(successo=True)

if __name__ == "__main__":
    main(parse_args())