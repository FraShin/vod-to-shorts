import subprocess
import os
import glob
import json
import sys

import config
from subtitles import generate_clip_ass, load_words_from_transcription

OUTPUT_DIR = os.path.join(os.path.dirname(config.SCRIPTS_DIR), "Clips_Manuali")
TRANSCRIPTION_FILE = os.path.join(config.OUTPUT_DIR, "transcription.json")

# Timeout e argomenti del codec arrivano da config: una sola fonte per FFmpeg,
# condivisa con cut_engine.py. USE_NVENC nel .env sceglie NVENC o libx264.


def _escape_ass_path(ass_path: str) -> str:
    """Path per filtro subtitles= di FFmpeg (slash + escape drive letter Windows)."""
    p = os.path.abspath(ass_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = p.replace(":", "\\:", 1)
    return p


def transcription_matches_vod(transcription_path: str, video_path: str) -> bool:
    if not os.path.exists(transcription_path):
        return False
    try:
        with open(transcription_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return False
    src = data.get("source_vod") or os.path.basename(data.get("source_vod_path", ""))
    if not src:
        return True  # vecchi JSON senza metadata: comportamento legacy
    return os.path.basename(video_path) == src


def parse_time(time_str):
    parts = list(map(int, time_str.split(":")))
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def get_latest_vods(limit=5):
    search_pattern = os.path.join(config.INPUT_DIR, "*.mp4")
    files = glob.glob(search_pattern)
    if not files:
        return []
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:limit]


def render_manual_clip(
    input_video: str,
    t_start: float,
    t_end: float,
    output_file: str,
    all_words: list,
) -> None:
    """
    Stesso schema di cut_engine:
    dual seek → encode verticale senza sub → ASS da VOD words → burn-in.
    """
    duration = t_end - t_start
    if duration <= 0:
        raise ValueError("Fine deve essere dopo l'inizio.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    temp_video = os.path.join(OUTPUT_DIR, "temp_manual_nosub.mp4")
    ass_path = os.path.join(OUTPUT_DIR, "temp_manual.ass")
    timeout = config.ffmpeg_timeout(duration)

    v_stack = config.ffmpeg_vertical_stack_filter()
    a_mix = config.ffmpeg_mix_audio_filter()

    seek_pre = max(0.0, float(t_start) - 2.0)
    seek_fine = max(0.0, float(t_start) - seek_pre)

    cmd_step1 = [
        config.FFMPEG_BIN, "-y",
        *config.ffmpeg_input_args(),
        "-ss", str(seek_pre),
        "-i", input_video,
        "-ss", str(seek_fine),
        "-t", str(duration),
        "-filter_complex", f"{v_stack};{a_mix}",
        "-map", "[v_stack]",
        "-map", "[a]",
        *config.ffmpeg_video_encoder_args(),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-r", str(config.VIDEO_FPS),
        "-loglevel", "warning",
        temp_video,
    ]

    err = None
    try:
        print(f"   → encode verticale (senza sub, timeout {timeout}s)...")
        subprocess.run(cmd_step1, check=True, timeout=timeout)

        has_subs = False
        if all_words:
            generate_clip_ass(t_start, t_end, all_words, ass_path, sub_delay=0.0)
            has_subs = os.path.isfile(ass_path) and os.path.getsize(ass_path) > 200

        if has_subs:
            ass_for_filter = _escape_ass_path(ass_path)
            cmd_step2 = [
                config.FFMPEG_BIN, "-y",
                "-i", temp_video,
                "-vf", f"subtitles={ass_for_filter}",
                *config.ffmpeg_video_encoder_args(),
                "-pix_fmt", "yuv420p", "-c:a", "copy",
                "-loglevel", "warning",
                output_file,
            ]
            print("   → burn-in sottotitoli ASS...")
            subprocess.run(cmd_step2, check=True, timeout=timeout)
        else:
            # Nessun sub: il temp è già il prodotto finale
            if os.path.exists(output_file):
                os.remove(output_file)
            os.replace(temp_video, output_file)
            temp_video = None  # non cancellare: già rinominato
    except Exception:
        err = True
        raise
    finally:
        if err is None:
            for p in (ass_path, temp_video):
                if p and os.path.exists(p):
                    os.remove(p)
        else:
            # Lascia temp/ass per debug se fallisce
            print(f"   ⚠️ Debug lasciato: {ass_path} / {temp_video}")


def main():
    print("=" * 50)
    print("     🎯 MANUAL CLIPPER v5 — dual seek + 2-pass (come cut_engine)")
    print("=" * 50)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    recent_vods = get_latest_vods()
    if not recent_vods:
        print(f"❌ Nessun video in {config.INPUT_DIR}/")
        return

    print("\n📂 Ultimi VOD:")
    for i, vod in enumerate(recent_vods):
        print(f"   [{i+1}] {os.path.basename(vod)}")

    choice = input("\n👉 Quale VOD? [1]: ").strip()
    vod_index = int(choice) - 1 if choice and choice.isdigit() else 0
    if vod_index < 0 or vod_index >= len(recent_vods):
        print("❌ Scelta non valida.")
        return
    input_video = recent_vods[vod_index]
    print(f"\n🎬 Editing: {os.path.basename(input_video)}")

    all_words = []
    if transcription_matches_vod(TRANSCRIPTION_FILE, input_video):
        all_words = load_words_from_transcription(TRANSCRIPTION_FILE)
        if all_words:
            print(
                f"✅ Sottotitoli da trascrizione ({len(all_words)} parole) "
                f"| lag={config.SUB_SYNC_LAG_SEC}s chunk≤{config.SUB_CHUNK_MAX_WORDS}"
            )
    else:
        print(
            "⚠️  Trascrizione non corrisponde a questo VOD (o assente). "
            "Clip SENZA sub. Lancia la pipeline sul VOD scelto, poi riprova."
        )

    while True:
        print("\n--- TAGLIO MANUALE ---")
        start_str = input("👉 Inizio (es 1:10, exit): ")
        if start_str.lower() == "exit":
            break
        end_str = input("👉 Fine (es 1:40):   ")
        title = input("👉 Titolo:       ")

        try:
            t_start = parse_time(start_str)
            t_end = parse_time(end_str)
            if t_end <= t_start:
                print("⚠️ Fine deve essere dopo l'inizio.")
                continue

            safe_title = config.clean_filename(title)
            output_file = os.path.join(OUTPUT_DIR, f"{safe_title}.mp4")

            print("⏳ Rendering...")
            render_manual_clip(input_video, t_start, t_end, output_file, all_words)
            print(f"✅ Salvata: {output_file}")

        except subprocess.TimeoutExpired:
            print("❌ FFmpeg timeout — clip troppo lunga o GPU bloccata.")
        except Exception as e:
            print(f"❌ Errore: {e}")


if __name__ == "__main__":
    main()
