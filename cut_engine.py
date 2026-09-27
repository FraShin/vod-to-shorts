import json
import os
import subprocess
import sys

import config
from subtitles import generate_clip_ass, load_words_from_transcription
from subtitles_clip import generate_ass_clip_local

try:
    from transcribe_engine import transcribe_clip_wav
except ImportError:
    transcribe_clip_wav = None

INPUT_CLIPS = os.path.join(config.OUTPUT_DIR, "final_choice.json")
TRANSCRIPTION_FILE = os.path.join(config.OUTPUT_DIR, "transcription.json")
CLIPS_DIR = config.CLIPS_OUTPUT_FOLDER

# Timeout e argomenti del codec arrivano da config: una sola fonte per FFmpeg,
# condivisa con manual_cutter.py. USE_NVENC nel .env sceglie NVENC o libx264.


def _run_ffmpeg(cmd: list[str], label: str, timeout: int) -> None:
    print(f"      → {label} (timeout {timeout}s)...", flush=True)
    subprocess.run(cmd, check=True, timeout=timeout)


def _extract_clip_wav(video_path: str, wav_path: str, timeout: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-loglevel", "warning",
        wav_path,
    ]
    _run_ffmpeg(cmd, "estrazione audio clip per Whisper", timeout)


def main():
    use_clip_words = config.SUB_WORD_SOURCE == "clip"
    print("📱 PRODUZIONE CLIP (v8 — sub da audio clip)" if use_clip_words else "📱 PRODUZIONE CLIP (v8 — sub da VOD json)")

    config.refresh_vod()
    input_video = config.VIDEO_FILENAME

    if not input_video or not os.path.exists(input_video):
        print(f"❌ VOD non trovato: {input_video}")
        return False

    if not os.path.exists(INPUT_CLIPS):
        print("❌ Manca final_choice.json.")
        return False

    if not use_clip_words and not os.path.exists(TRANSCRIPTION_FILE):
        print("❌ Manca transcription.json (o imposta SUB_WORD_SOURCE=clip).")
        return False

    if use_clip_words and transcribe_clip_wav is None:
        print("❌ transcribe_engine non disponibile per SUB_WORD_SOURCE=clip.")
        return False

    os.makedirs(CLIPS_DIR, exist_ok=True)

    with open(INPUT_CLIPS, "r", encoding="utf-8") as f:
        clips = json.load(f)

    if not clips:
        print("❌ Nessuna clip in final_choice.json.")
        return False

    all_words = []
    if not use_clip_words:
        all_words = load_words_from_transcription(TRANSCRIPTION_FILE)

    print(
        f"   📝 Sub: source={config.SUB_WORD_SOURCE} "
        f"lag={config.SUB_SYNC_LAG_SEC}s chunk≤{config.SUB_CHUNK_MAX_WORDS} "
        f"gap={config.SUB_CHUNK_GAP_SEC}s chars≤{config.SUB_CHUNK_MAX_CHARS} "
        f"style={config.SUB_MULTI_WORD_STYLE}"
    )

    failed = 0
    rendered = 0
    skipped = 0

    v_filter = config.ffmpeg_vertical_stack_filter()
    a_filter = config.ffmpeg_mix_audio_filter()

    for i, clip in enumerate(clips):
        start = clip["start"]
        end = clip["end"]
        duration = end - start

        safe_text = config.clean_filename(clip.get("text", ""))
        output_path = os.path.join(CLIPS_DIR, f"Clip_{i+1}_{safe_text}.mp4")
        temp_no_subs = os.path.join(CLIPS_DIR, f"temp_nosub_{i}.mp4")
        temp_wav = os.path.join(CLIPS_DIR, f"temp_clip_{i}.wav")
        ass_path = os.path.join(CLIPS_DIR, f"temp_sub_{i}.ass")

        if os.path.isfile(output_path) and os.path.getsize(output_path) > 100_000:
            print(f"   ⏭️  Clip {i+1}/{len(clips)} già presente, skip.")
            skipped += 1
            continue

        print(f"   🎬 Clip {i+1}/{len(clips)}: {safe_text} ({duration:.1f}s)")

        ass_for_filter = ""
        timeout = config.ffmpeg_timeout(duration)

        seek_pre = max(0.0, float(start) - 2.0)
        seek_fine = max(0.0, float(start) - seek_pre)

        cmd_step1 = [
            "ffmpeg", "-y",
            *config.ffmpeg_input_args(),
            "-ss", str(seek_pre),
            "-i", input_video,
            "-ss", str(seek_fine),
            "-t", str(duration),
            "-filter_complex",
            f"{v_filter};{a_filter}",
            "-map", "[v_stack]",
            "-map", "[a]",
            *config.ffmpeg_video_encoder_args(),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-r", str(config.VIDEO_FPS),
            "-loglevel", "warning",
            temp_no_subs,
        ]

        err = None
        try:
            _run_ffmpeg(cmd_step1, "encode verticale (senza sub)", timeout)

            if use_clip_words:
                _extract_clip_wav(temp_no_subs, temp_wav, timeout=min(timeout, 120))
                print("      → Whisper sulla clip...", flush=True)
                segs = transcribe_clip_wav(temp_wav)
                print(f"      → {len(segs)} segmenti Whisper sulla clip", flush=True)
                generate_ass_clip_local(duration, segs, ass_path)
            else:
                generate_clip_ass(start, end, all_words, ass_path, sub_delay=0.0)

            ass_for_filter = os.path.abspath(ass_path).replace("\\", "/")
            if len(ass_for_filter) >= 2 and ass_for_filter[1] == ":":
                ass_for_filter = ass_for_filter.replace(":", "\\:", 1)

            cmd_step2 = [
                "ffmpeg", "-y",
                "-i", temp_no_subs,
                "-vf", f"subtitles={ass_for_filter}",
                *config.ffmpeg_video_encoder_args(),
                "-pix_fmt", "yuv420p", "-c:a", "copy",
                "-loglevel", "warning",
                output_path,
            ]
            _run_ffmpeg(cmd_step2, "burn-in sottotitoli ASS", timeout)
            rendered += 1
        except subprocess.TimeoutExpired:
            failed += 1
            err = "timeout"
            print(f"   ❌ Clip {i+1}: FFmpeg troppo lento / bloccato ({err}).")
            print(f"      Lasciato per debug: {ass_path}")
        except subprocess.CalledProcessError as e:
            failed += 1
            err = str(e)
            print(f"   ❌ Clip {i+1}: {e}")
            print(f"      Lasciato per debug: {ass_path}")
        except Exception as e:
            failed += 1
            err = str(e)
            print(f"   ❌ Clip {i+1}: {e}")
            print(f"      Lasciato per debug: {ass_path}")
        finally:
            if err is None:
                for p in (ass_path, temp_no_subs, temp_wav):
                    if os.path.exists(p):
                        os.remove(p)

    print(f"✨ Fine: {rendered} nuove, {skipped} skip, {failed} errori → {CLIPS_DIR}")
    return failed == 0 and (rendered + skipped) > 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)