import base64
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

# Load .env BEFORE importing config, so environment variables are available
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

import requests

from app import config
from app.prompts import (
    SCENE_EXPLAIN_SYSTEM_PROMPT,
    SCENE_SUMMARY_SYSTEM_PROMPT,
    SCENE_GROUNDED_SUMMARY_SYSTEM_PROMPT,
    SCENE_TRANSCRIPT_ENHANCEMENT_SYSTEM_PROMPT,
    build_scene_explain_user_text,
    build_scene_summary_user_text,
    build_scene_grounded_summary_user_text,
    build_transcript_enhancement_user_text,
    format_video_context_for_prompt,
)
from app.transcription_provider import get_transcriber
from app.video_context import (
    analyze_video_for_context,
    save_video_context,
    load_video_context,
)

UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
STATIC = ROOT / "static"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)


def _which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def _probe_duration_seconds(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not (result.stdout or "").strip():
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _has_audio_stream(video_path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool((result.stdout or "").strip())


def _detect_shots(video_path: Path) -> list[tuple[float, float]]:
    """Shot boundaries from PySceneDetect: each tuple is (start_sec, end_sec) for one shot."""
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=config.SCENE_DETECTION_THRESHOLD))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    out: list[tuple[float, float]] = []
    for start_tc, end_tc in scenes:
        start_s = start_tc.get_seconds()
        end_s = end_tc.get_seconds()
        if end_s > start_s:
            out.append((start_s, end_s))
    return out


def _split_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    out_path: Path,
    ffmpeg: str,
    *,
    has_audio: bool,
) -> None:
    if end_sec <= start_sec:
        raise ValueError("invalid segment duration")
    # Re-encode (not stream copy) so cuts match the detected timestamps. With -c copy,
    # FFmpeg aligns to keyframes and the file often starts at an earlier keyframe —
    # players then show black or junk for seconds until the next keyframe.
    cmd: List[str] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{start_sec:.6f}",
        "-to",
        f"{end_sec:.6f}",
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        cmd.extend(
            ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"],
        )
    cmd.extend(["-movflags", "+faststart", str(out_path)])
    subprocess.run(cmd, check=True)


def _chunk_shots(
    shots: List[Tuple[float, float]],
    shots_per_scene: int,
) -> List[List[Tuple[float, float]]]:
    if shots_per_scene < 1:
        raise ValueError("shots_per_scene must be >= 1")
    return [shots[i : i + shots_per_scene] for i in range(0, len(shots), shots_per_scene)]


def _concat_shot_files(
    ffmpeg: str,
    shot_paths: List[Path],
    out_path: Path,
) -> None:
    if not shot_paths:
        raise ValueError("no shots to concatenate")
    if len(shot_paths) == 1:
        shutil.copy2(shot_paths[0], out_path)
        return

    shots_dir = shot_paths[0].parent
    for p in shot_paths:
        if p.parent != shots_dir:
            raise ValueError("all shots must share the same folder")

    list_file = shots_dir / "_concat_list.txt"
    try:
        with list_file.open("w", encoding="utf-8") as f:
            for p in shot_paths:
                f.write(f"file '{p.name}'\n")
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        if list_file.exists():
            list_file.unlink()


def _extract_scene_audio(
    ffmpeg: str,
    scene_video: Path,
    audio_out: Path,
) -> None:
    cmd: list[str] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(scene_video),
        "-vn",
        "-map",
        "0:a:0",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(audio_out),
    ]
    subprocess.run(cmd, check=True)


def _extract_full_audio_wav(
    ffmpeg: str,
    video_file: Path,
    audio_out: Path,
) -> None:
    """Extract full video audio to WAV format (preserving original quality)."""
    cmd: list[str] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_file),
        "-vn",
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_s16le",
        str(audio_out),
    ]
    subprocess.run(cmd, check=True)


def _transcript_paths_sorted(transcripts_dir: Path) -> List[Path]:
    files = list(transcripts_dir.glob("scene_*.json"))

    def sort_key(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    return sorted(files, key=sort_key)


def _build_full_transcript_context(transcripts_dir: Path, max_chars: int) -> str:
    parts: List[str] = []
    for p in _transcript_paths_sorted(transcripts_dir):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        idx = data.get("scene_index") or p.stem
        txt = (data.get("text") or "").strip()
        if not txt:
            continue
        segs = data.get("segments") or []
        if segs:

            def line(s: dict) -> str:
                sp = s.get("speaker")
                t = (s.get("text") or "").strip()
                return f'[{sp}] {t}' if sp else t

            line_txt = "\n".join(line(s) for s in segs if (s.get("text") or "").strip())
            parts.append(f"=== Scene {idx} ===\n{line_txt}")
        else:
            parts.append(f"=== Scene {idx} ===\n{txt}")
    full = "\n\n".join(parts)
    if not full.strip():
        return "(No transcript text in any scene.)"
    if len(full) <= max_chars:
        return full
    return full[: max_chars - 30] + "\n\n[… transcript truncated …]"


def _scene_transcript_block(transcript_path: Optional[Path]) -> str:
    if transcript_path is None or not transcript_path.is_file():
        return "(No transcript available for this scene.)"
    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "(Transcript file could not be read.)"
    return _format_transcript_segments(data)


def _format_transcript_segments(transcript_data: dict) -> str:
    """Format transcript dict into readable text with timestamps and speakers."""
    segs = transcript_data.get("segments") or []
    lines: List[str] = []
    for s in segs:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        sp = s.get("speaker")
        st = float(s.get("start") or 0)
        en = float(s.get("end") or 0)
        sp_part = f"[{sp}] " if sp else ""
        lines.append(f"[{st:.1f}s–{en:.1f}s] {sp_part}{t}")
    if lines:
        return "\n".join(lines)
    return (transcript_data.get("text") or "").strip() or "(Empty transcript.)"


def _extract_scene_frames_jpeg(
    ffmpeg: str,
    video_path: Path,
    out_dir: Path,
    *,
    interval_ms: int,
    max_frames: int,
) -> List[Tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = 1000.0 / float(max(200, min(interval_ms, 5000)))
    pattern = str(out_dir / "frame_%04d.jpg")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps:.6f}",
            "-frames:v",
            str(max_frames),
            pattern,
        ],
        check=True,
    )
    files = sorted(out_dir.glob("frame_*.jpg"))
    out: List[Tuple[float, Path]] = []
    step_s = interval_ms / 1000.0
    for i, p in enumerate(files):
        out.append((i * step_s, p))
    if out:
        return out
    single = out_dir / "frame_fallback.jpg"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "0",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-vf",
            "scale=960:-1",
            str(single),
        ],
        check=True,
    )
    if single.is_file():
        return [(0.0, single)]
    return []


def _jpeg_b64_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _openai_explain_scene(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_content: List[Dict],
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "max_tokens": config.OPENAI_EXPLAIN_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SCENE_EXPLAIN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=config.OPENAI_EXPLAIN_TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(r.text)
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(str(data))
    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _get_scene_duration_seconds(
    transcript_path: Optional[Path],
    video_path: Path,
) -> Optional[float]:
    """Extract scene duration from transcript metadata, or from video file.

    Prefers transcript metadata (already computed, no file I/O). Falls back to
    probing the video file if transcript doesn't contain timeline information.

    Args:
        transcript_path: Path to scene_NNN.json (contains timeline_in_source_video)
        video_path: Path to scene_NNN.mp4 (fallback if transcript unavailable)

    Returns:
        Duration in seconds (rounded to 3 decimals), or None if unavailable
    """
    if transcript_path and transcript_path.is_file():
        try:
            data = json.loads(transcript_path.read_text(encoding="utf-8"))
            timeline = data.get("timeline_in_source_video")
            if timeline and "start" in timeline and "end" in timeline:
                return round(timeline["end"] - timeline["start"], 3)
        except (OSError, json.JSONDecodeError):
            pass

    if video_path.is_file():
        duration = _probe_duration_seconds(video_path)
        return duration if duration > 0 else None

    return None


def _openai_summarize_explanation(
    *,
    base_url: str,
    api_key: str,
    model: str,
    explanation_text: str,
    duration_seconds: float,
) -> str:
    """Generate a time-constrained summary of a scene explanation.

    Summarizes the explanation to fit within the scene's audio duration,
    calculated at ~150 words per minute (2.3 words/second for natural pacing).

    Args:
        base_url: OpenAI API base URL
        api_key: OpenAI API key
        model: Model to use for summarization
        explanation_text: Full explanation to summarize
        duration_seconds: Audio duration constraint (in seconds)

    Returns:
        Summarized explanation text

    Raises:
        RuntimeError: If API request fails
    """
    target_words = max(
        config.SUMMARY_MIN_WORDS,
        int(duration_seconds * config.SUMMARY_WORDS_PER_SECOND),
    )
    target_words = min(target_words, config.SUMMARY_MAX_WORDS)

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "max_tokens": config.OPENAI_SUMMARY_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SCENE_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_scene_summary_user_text(
                    explanation=explanation_text,
                    duration_seconds=duration_seconds,
                    target_word_count=target_words,
                ),
            },
        ],
    }

    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=config.OPENAI_SUMMARY_TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(r.text)

    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in summary response: {data}")

    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _openai_generate_grounded_summary(
    *,
    base_url: str,
    api_key: str,
    model: str,
    explanation_text: str,
    scene_transcript: str,
    duration_seconds: float,
) -> str:
    """Generate a grounded summary using actual words from the scene transcript.

    Creates a conversational summary that uses at least 50% of its words from the
    actual video transcript, as if a friend is narrating what happened.

    Args:
        base_url: OpenAI API base URL
        api_key: OpenAI API key
        model: Model to use for summarization
        explanation_text: Full explanation (for context)
        scene_transcript: Actual dialogue from the scene (for word grounding)
        duration_seconds: Audio duration constraint (in seconds)

    Returns:
        Grounded summary text using actual transcript words

    Raises:
        RuntimeError: If API request fails
    """
    target_words = max(
        config.SUMMARY_MIN_WORDS,
        int(duration_seconds * config.SUMMARY_WORDS_PER_SECOND),
    )
    target_words = min(target_words, config.SUMMARY_MAX_WORDS)

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "max_tokens": config.OPENAI_GROUNDED_SUMMARY_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SCENE_GROUNDED_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_scene_grounded_summary_user_text(
                    explanation=explanation_text,
                    scene_transcript=scene_transcript,
                    duration_seconds=duration_seconds,
                    target_word_count=target_words,
                ),
            },
        ],
    }

    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=config.OPENAI_GROUNDED_SUMMARY_TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(r.text)

    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in grounded summary response: {data}")

    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _openai_enhance_scene_transcript(
    *,
    base_url: str,
    api_key: str,
    model: str,
    scene_transcript: str,
    grounded_summary: str,
    video_context_text: str = "",
) -> str:
    """Enhance scene transcript with visual/contextual details using a second AI model.

    Adds speaker identification, objects, actions, and scene context from grounded summary
    and video context to make the transcript more informative for narration audiences.

    Args:
        base_url: OpenAI API base URL
        api_key: OpenAI API key
        model: Model to use for enhancement
        scene_transcript: Scene transcript with timestamps and speakers
        grounded_summary: Grounded summary text from visual analysis
        video_context_text: Character and object detection data (optional)

    Returns:
        Enhanced transcript text with added contextual details

    Raises:
        RuntimeError: If API request fails
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "max_tokens": 2000,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": SCENE_TRANSCRIPT_ENHANCEMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_transcript_enhancement_user_text(
                    scene_transcript=scene_transcript,
                    grounded_summary=grounded_summary,
                    video_context_text=video_context_text,
                ),
            },
        ],
    }

    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=config.OPENAI_GROUNDED_SUMMARY_TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(r.text)

    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in enhancement response: {data}")

    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _tts_input_clip(text: str) -> str:
    limit = config.OPENAI_TTS_MAX_INPUT_CHARS
    t = text.strip()
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 1)] + "…"


def _openai_tts_to_mp3(
    *,
    base_url: str,
    api_key: str,
    text: str,
    out_path: Path,
    voice: Optional[str] = None,
    speed: float = 1.0,
) -> None:
    url = f"{base_url.rstrip('/')}/audio/speech"
    payload = {
        "model": config.OPENAI_TTS_MODEL,
        "voice": voice or config.OPENAI_TTS_VOICE,
        "input": _tts_input_clip(text),
        "response_format": "mp3",
        "speed": speed,
    }
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=config.OPENAI_TTS_TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(r.text)
    out_path.write_bytes(r.content)


def _try_write_explanation_narration(
    *,
    base_url: str,
    api_key: str,
    explanation_text: str,
    mp3_path: Path,
) -> bool:
    try:
        _openai_tts_to_mp3(
            base_url=base_url,
            api_key=api_key,
            text=explanation_text,
            out_path=mp3_path,
        )
    except Exception:
        return False
    return mp3_path.is_file()


app = FastAPI(title="Scene clip test")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    index_file = STATIC / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    raise HTTPException(status_code=404)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    ext = Path(file.filename).suffix or ".mp4"
    uid = str(uuid.uuid4())
    dest = UPLOADS / f"{uid}{ext}"
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"id": uid, "filename": file.filename, "path_suffix": dest.suffix}


@app.post("/api/process/{video_id}")
def process(
    video_id: str,
    shots_per_scene: int = Query(
        default=config.DEFAULT_SHOTS_PER_SCENE,
        ge=config.MIN_SHOTS_PER_SCENE,
        le=config.MAX_SHOTS_PER_SCENE,
    ),
    provider: Optional[str] = Query(
        default=None,
        description="Transcription provider: whisper or assemblyai. Defaults to config value.",
    ),
):
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found on PATH. Install ffmpeg and retry.",
        )

    matches = list(UPLOADS.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Upload not found")
    video_path = matches[0]

    session_dir = OUTPUTS / video_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True)
    shots_dir = session_dir / "shots"
    scenes_dir = session_dir / "scenes"
    scene_audio_dir = session_dir / "scene_audio"
    transcripts_dir = session_dir / "transcripts"
    shots_dir.mkdir()
    scenes_dir.mkdir()
    scene_audio_dir.mkdir()
    transcripts_dir.mkdir()

    try:
        ranges = _detect_shots(video_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shot detection failed: {e}") from e

    if not ranges:
        duration = _probe_duration_seconds(video_path)
        if duration > 0:
            ranges = [(0.0, duration)]

    if not ranges:
        raise HTTPException(
            status_code=422,
            detail="No shots detected and duration could not be read.",
        )

    has_audio = _has_audio_stream(video_path)

    # Extract full video audio and transcribe once for entire video
    full_transcript = None
    if has_audio:
        try:
            full_audio_path = session_dir / "audio.wav"
            _extract_full_audio_wav(ffmpeg, video_path, full_audio_path)

            # Transcribe full audio once
            try:
                selected_provider = provider or config.TRANSCRIPTION_PROVIDER
                transcriber = get_transcriber(selected_provider)
            except ValueError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e

            try:
                result = transcriber.transcribe(full_audio_path)
                full_transcript = result.to_dict()
                full_transcript["provider_used"] = result.provider

                # Save full video transcript
                provider_suffix = result.provider if result.provider else "whisper"
                full_transcript_path = transcripts_dir / f"full_video_{provider_suffix}.json"
                full_transcript_path.write_text(
                    json.dumps(full_transcript, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"✓ Full video transcribed with {result.provider} ({len(full_transcript.get('segments', []))} segments)")
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Full video transcription failed: {e}",
                ) from e
        except HTTPException:
            raise
        except Exception as e:
            # Non-blocking - continue processing even if audio extraction/transcription fails
            print(f"Warning: Failed to transcribe full video for {video_id}: {e}")

    shot_entries: List[Dict] = []
    shot_paths_ordered: List[Path] = []

    for i, (start_s, end_s) in enumerate(ranges, start=1):
        clip_name = f"shot_{i:03d}.mp4"
        out_file = shots_dir / clip_name
        try:
            _split_clip(
                video_path,
                start_s,
                end_s,
                out_file,
                ffmpeg,
                has_audio=has_audio,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Split failed for shot {i}: {e}",
            ) from e
        dur = round(end_s - start_s, 3)
        shot_paths_ordered.append(out_file)
        shot_entries.append(
            {
                "index": i,
                "start": round(start_s, 3),
                "end": round(end_s, 3),
                "duration": dur,
                "filename": clip_name,
                "url": f"/api/clips/{video_id}/shots/{clip_name}",
            },
        )

    scene_groups = _chunk_shots(ranges, shots_per_scene)
    scene_entries: List[Dict] = []
    shot_offset = 0

    for si, group in enumerate(scene_groups, start=1):
        start_s = group[0][0]
        end_s = group[-1][1]
        dur = round(end_s - start_s, 3)
        shot_indices = list(
            range(shot_offset + 1, shot_offset + len(group) + 1),
        )
        paths_for_scene = shot_paths_ordered[
            shot_offset : shot_offset + len(group)
        ]
        shot_offset += len(group)
        scene_name = f"scene_{si:03d}.mp4"
        out_scene = scenes_dir / scene_name
        try:
            _concat_shot_files(ffmpeg, paths_for_scene, out_scene)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to merge scene {si}: {e}",
            ) from e

        entry: dict = {
            "index": si,
            "start": round(start_s, 3),
            "end": round(end_s, 3),
            "duration": dur,
            "shot_indices": shot_indices,
            "filename": scene_name,
            "url": f"/api/clips/{video_id}/scenes/{scene_name}",
            "audio_url": None,
            "audio_filename": None,
            "transcript_url": None,
            "transcript_filename": None,
        }
        if has_audio and _has_audio_stream(out_scene):
            audio_name = f"scene_{si:03d}.m4a"
            out_audio = scene_audio_dir / audio_name
            try:
                _extract_scene_audio(ffmpeg, out_scene, out_audio)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to extract audio for scene {si}: {e}",
                ) from e
            entry["audio_filename"] = audio_name
            entry["audio_url"] = f"/api/clips/{video_id}/scene_audio/{audio_name}"

            # Use full transcript if available, extract segments for this scene
            if full_transcript:
                provider_suffix = full_transcript.get("provider_used", "whisper")

                # Filter transcript segments for this scene's time range
                scene_segments = [
                    seg for seg in full_transcript.get("segments", [])
                    if seg.get("start", 0) >= start_s and seg.get("end", float('inf')) <= end_s
                ]

                # Create scene-specific transcript with full context
                scene_payload = {
                    "scene_index": si,
                    "audio_filename": audio_name,
                    "timeline_in_source_video": {
                        "start": round(start_s, 3),
                        "end": round(end_s, 3),
                    },
                    "provider": provider_suffix,
                    "language": full_transcript.get("language"),
                    "language_probability": full_transcript.get("language_probability"),
                    "diarization": full_transcript.get("diarization"),
                    "segments": scene_segments,
                    "text": " ".join([seg.get("text", "") for seg in scene_segments]).strip(),
                    "full_video_transcript_file": f"full_video_{provider_suffix}.json",
                }

                transcript_name = f"scene_{si:03d}_{provider_suffix}.json"
                transcript_path = transcripts_dir / transcript_name
                transcript_path.write_text(
                    json.dumps(scene_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                entry["transcript_filename"] = transcript_name
                entry["transcript_url"] = f"/api/transcript/{video_id}/{transcript_name}"

        scene_entries.append(entry)

    # Automatically trigger video context analysis after processing
    # (don't fail if it fails - it's optional)
    try:
        context = analyze_video_for_context(
            video_path,
            frame_interval_ms=config.SPARSE_FRAME_INTERVAL_MS,
        )
        if context:
            save_video_context(context, video_id, OUTPUTS)
    except Exception as e:
        # Log but don't fail - video context is optional for scene explanations
        print(f"Warning: Video context analysis failed for {video_id}: {e}")

    return {
        "video_id": video_id,
        "shots_per_scene": shots_per_scene,
        "provider": provider or config.TRANSCRIPTION_PROVIDER,
        "shots": shot_entries,
        "scenes": scene_entries,
    }


@app.get("/api/list-outputs")
def list_outputs():
    """List all previously generated videos and their outputs."""
    if not OUTPUTS.is_dir():
        return JSONResponse({"videos": []})

    videos = []
    for video_dir in sorted(OUTPUTS.iterdir()):
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name
        shots_dir = video_dir / "shots"
        scenes_dir = video_dir / "scenes"

        shots = []
        scenes = []

        if shots_dir.is_dir():
            for shot_file in sorted(shots_dir.glob("shot_*.mp4")):
                match = shot_file.stem.split("_")
                if len(match) >= 2 and match[1].isdigit():
                    idx = int(match[1])
                    shots.append({
                        "index": idx,
                        "url": f"/api/clips/{video_id}/shots/{shot_file.name}",
                    })

        if scenes_dir.is_dir():
            for scene_file in sorted(scenes_dir.glob("scene_*.mp4")):
                match = scene_file.stem.split("_")
                if len(match) >= 2 and match[1].isdigit():
                    idx = int(match[1])
                    scene_audio_name = f"scene_{idx:03d}.m4a"
                    audio_path = video_dir / "scene_audio" / scene_audio_name

                    audio_url = None
                    if audio_path.is_file():
                        audio_url = f"/api/clips/{video_id}/scene_audio/{scene_audio_name}"

                    # Look for transcript files with provider suffix (e.g., scene_001_assemblyai.json)
                    # Prefer assemblyai over whisper if both exist
                    transcript_url = None
                    transcript_name = None

                    transcripts_dir = video_dir / "transcripts"
                    if transcripts_dir.is_dir():
                        # Check for AssemblyAI transcript first
                        assemblyai_transcript = f"scene_{idx:03d}_assemblyai.json"
                        assemblyai_path = transcripts_dir / assemblyai_transcript

                        if assemblyai_path.is_file():
                            transcript_name = assemblyai_transcript
                            transcript_url = f"/api/transcript/{video_id}/{assemblyai_transcript}"
                        else:
                            # Fall back to Whisper transcript
                            whisper_transcript = f"scene_{idx:03d}_whisper.json"
                            whisper_path = transcripts_dir / whisper_transcript

                            if whisper_path.is_file():
                                transcript_name = whisper_transcript
                                transcript_url = f"/api/transcript/{video_id}/{whisper_transcript}"

                    scenes.append({
                        "index": idx,
                        "url": f"/api/clips/{video_id}/scenes/{scene_file.name}",
                        "audio_url": audio_url,
                        "transcript_url": transcript_url,
                    })

        if shots or scenes:
            videos.append({
                "video_id": video_id,
                "shots": shots,
                "scenes": scenes,
            })

    return JSONResponse({"videos": videos})


@app.post("/api/clear")
def clear_all():
    """Delete all generated outputs (scenes, shots, transcripts, explanations)."""
    if OUTPUTS.is_dir():
        try:
            shutil.rmtree(OUTPUTS)
            OUTPUTS.mkdir(parents=True, exist_ok=True)
            return JSONResponse({"status": "ok", "message": "All outputs cleared"})
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear outputs: {e}",
            )
    return JSONResponse({"status": "ok", "message": "No outputs to clear"})


@app.get("/api/transcript/{video_id}/full-video")
def get_full_video_transcript(video_id: str):
    """Get the full video transcript with all speakers and timings."""
    transcripts_dir = OUTPUTS / video_id / "transcripts"

    if not transcripts_dir.is_dir():
        raise HTTPException(status_code=404, detail="No transcripts available")

    # Look for full_video transcript (prefer assemblyai over whisper)
    full_video_files = list(transcripts_dir.glob("full_video_*.json"))

    if not full_video_files:
        raise HTTPException(status_code=404, detail="Full video transcript not found")

    # Prefer assemblyai if both exist
    full_video_path = None
    for f in full_video_files:
        if "assemblyai" in f.name:
            full_video_path = f
            break

    if not full_video_path:
        full_video_path = full_video_files[0]

    try:
        data = json.loads(full_video_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid transcript file")

    return JSONResponse(data)


@app.get("/api/transcript/{video_id}/{filename}")
def get_transcript(video_id: str, filename: str):
    safe = Path(filename).name
    parts = safe.split(".")
    if (
        len(parts) != 2
        or parts[1] != "json"
        or not parts[0].startswith("scene_")
    ):
        raise HTTPException(status_code=404)
    path = OUTPUTS / video_id / "transcripts" / safe
    if not path.is_file():
        raise HTTPException(status_code=404)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid transcript file")
    return JSONResponse(data)


@app.post("/api/analyze-video/{video_id}")
def analyze_video_endpoint(video_id: str):
    """
    Analyze entire video for character and object continuity.

    This endpoint extracts sparse frames from the video, performs face
    detection/clustering and object detection, and caches the results
    for use in scene explanations.
    """
    session = OUTPUTS / video_id
    video_file = session / "input.mp4"

    if not video_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video file not found",
        )

    # Check if analysis is already cached
    existing_context = load_video_context(video_id, OUTPUTS)
    if existing_context is not None:
        return JSONResponse({
            "status": "cached",
            "message": "Video context already analyzed",
            "characters_count": len(existing_context.characters),
            "objects_count": len(existing_context.objects),
        })

    try:
        # Analyze video for context
        context = analyze_video_for_context(
            video_file,
            frame_interval_ms=config.SPARSE_FRAME_INTERVAL_MS,
        )

        if context is None:
            raise HTTPException(
                status_code=500,
                detail="Video analysis failed",
            )

        # Cache the context
        save_video_context(context, video_id, OUTPUTS)

        return JSONResponse({
            "status": "success",
            "message": "Video analysis completed",
            "characters_count": len(context.characters),
            "objects_count": len(context.objects),
            "characters": [
                {
                    "id": c.id,
                    "confidence": c.confidence,
                    "appearance_count": len(c.appearances),
                }
                for c in context.characters
            ],
            "objects": [
                {
                    "id": o.id,
                    "class": o.class_label,
                    "appearance_count": len(o.appearances),
                }
                for o in context.objects
            ],
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {e}",
        ) from e


@app.post("/api/explain-scene/{video_id}/{scene_index}")
def explain_scene(
    video_id: str,
    scene_index: int,
    frame_interval_ms: int = Query(
        default=config.EXPLAIN_FRAME_INTERVAL_MS_DEFAULT,
        ge=config.EXPLAIN_FRAME_INTERVAL_MS_MIN,
        le=config.EXPLAIN_FRAME_INTERVAL_MS_MAX,
    ),
    regenerate: bool = Query(default=False),
):
    if scene_index < 1:
        raise HTTPException(status_code=400, detail="scene_index must be >= 1")
    session = OUTPUTS / video_id
    scene_mp4 = session / "scenes" / f"scene_{scene_index:03d}.mp4"
    if not scene_mp4.is_file():
        raise HTTPException(status_code=404, detail="Scene clip not found")

    explanations_dir = session / "explanations"
    explanations_dir.mkdir(parents=True, exist_ok=True)
    cache_path = explanations_dir / f"scene_{scene_index:03d}.txt"
    audio_name = f"scene_{scene_index:03d}.mp3"
    audio_path = explanations_dir / audio_name

    api_key = config.OPENAI_API_KEY
    base_url = config.OPENAI_BASE_URL

    if cache_path.is_file() and not regenerate:
        explanation_text = cache_path.read_text(encoding="utf-8")

        summary_cache_path = explanations_dir / f"scene_{scene_index:03d}_summary.txt"
        summary_text = None
        if summary_cache_path.is_file():
            summary_text = summary_cache_path.read_text(encoding="utf-8")

        grounded_cache_path = explanations_dir / f"scene_{scene_index:03d}_grounded.txt"
        grounded_summary_text = None
        if grounded_cache_path.is_file():
            grounded_summary_text = grounded_cache_path.read_text(encoding="utf-8")

        transcripts_dir = session / "transcripts"
        tp = transcripts_dir / f"scene_{scene_index:03d}.json"
        duration_seconds = _get_scene_duration_seconds(
            tp if tp.is_file() else None,
            scene_mp4,
        )

        explanation_audio_url: Optional[str] = None
        if api_key and not audio_path.is_file():
            _try_write_explanation_narration(
                base_url=base_url,
                api_key=api_key,
                explanation_text=explanation_text,
                mp3_path=audio_path,
            )
        if audio_path.is_file():
            explanation_audio_url = f"/api/explanations/{video_id}/{audio_name}"

        summary_audio_url: Optional[str] = None
        summary_audio_name = f"scene_{scene_index:03d}_summary.mp3"
        summary_audio_path = explanations_dir / summary_audio_name

        if summary_text and api_key and not summary_audio_path.is_file():
            try:
                _openai_tts_to_mp3(
                    base_url=base_url,
                    api_key=api_key,
                    text=summary_text,
                    out_path=summary_audio_path,
                )
            except Exception:
                pass

        if summary_audio_path.is_file():
            summary_audio_url = f"/api/explanations/{video_id}/{summary_audio_name}"

        grounded_audio_url: Optional[str] = None
        grounded_audio_name = f"scene_{scene_index:03d}_grounded.mp3"
        grounded_audio_path = explanations_dir / grounded_audio_name

        if grounded_summary_text and api_key and not grounded_audio_path.is_file():
            try:
                _openai_tts_to_mp3(
                    base_url=base_url,
                    api_key=api_key,
                    text=grounded_summary_text,
                    out_path=grounded_audio_path,
                )
            except Exception:
                pass

        if grounded_audio_path.is_file():
            grounded_audio_url = f"/api/explanations/{video_id}/{grounded_audio_name}"

        return JSONResponse(
            {
                "explanation": explanation_text,
                "summary": summary_text,
                "grounded_summary": grounded_summary_text,
                "duration_seconds": duration_seconds,
                "cached": True,
                "scene_index": scene_index,
                "explanation_audio_url": explanation_audio_url,
                "summary_audio_url": summary_audio_url,
                "grounded_summary_audio_url": grounded_audio_url,
            },
        )

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Set OPENAI_API_KEY in a .env file at the project root or in the environment.",
        )

    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found on PATH",
        )

    model = config.OPENAI_EXPLAIN_MODEL
    max_frames = config.EXPLAIN_MAX_FRAMES
    max_context = config.EXPLAIN_FULL_TRANSCRIPT_MAX_CHARS

    transcripts_dir = session / "transcripts"
    full_text = (
        _build_full_transcript_context(transcripts_dir, max_context)
        if transcripts_dir.is_dir()
        else "(No transcripts available for this video.)"
    )

    tp = transcripts_dir / f"scene_{scene_index:03d}.json"
    scene_txt = _scene_transcript_block(tp if tp.is_file() else None)

    with tempfile.TemporaryDirectory(prefix="test-clips-frames-") as tmp:
        tmp_path = Path(tmp)
        frames = _extract_scene_frames_jpeg(
            ffmpeg,
            scene_mp4,
            tmp_path,
            interval_ms=frame_interval_ms,
            max_frames=max_frames,
        )
        if not frames:
            raise HTTPException(
                status_code=500,
                detail="Could not extract frames from scene video",
            )

        # Load video context if available
        video_context = load_video_context(video_id, OUTPUTS)
        video_context_text = ""
        if video_context:
            video_context_text = format_video_context_for_prompt(
                video_context,
                character_confidence_threshold=config.CHARACTER_CONFIDENCE_THRESHOLD_PERCENT,
            )

        user_content: List[Dict] = [
            {
                "type": "text",
                "text": build_scene_explain_user_text(
                    full_transcript=full_text,
                    scene_transcript=scene_txt,
                    frame_count=len(frames),
                    frame_interval_ms=frame_interval_ms,
                    video_context_text=video_context_text,
                ),
            },
        ]
        img_detail = config.OPENAI_IMAGE_DETAIL
        for _t_sec, fp in frames:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _jpeg_b64_data_url(fp),
                        "detail": img_detail,
                    },
                },
            )

        try:
            explanation = _openai_explain_scene(
                base_url=base_url,
                api_key=api_key,
                model=model,
                user_content=user_content,
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"AI request failed: {e}",
            ) from e

    if not explanation:
        raise HTTPException(
            status_code=502,
            detail="Model returned an empty explanation",
        )

    cache_path.write_text(explanation, encoding="utf-8")

    summary_cache_path = explanations_dir / f"scene_{scene_index:03d}_summary.txt"
    summary_text = None
    duration_seconds = None

    tp = transcripts_dir / f"scene_{scene_index:03d}.json"
    duration_seconds = _get_scene_duration_seconds(
        tp if tp.is_file() else None,
        scene_mp4,
    )

    if duration_seconds:
        try:
            summary_text = _openai_summarize_explanation(
                base_url=base_url,
                api_key=api_key,
                model=model,
                explanation_text=explanation,
                duration_seconds=duration_seconds,
            )
            summary_cache_path.write_text(summary_text, encoding="utf-8")
        except Exception:
            summary_text = None

    grounded_cache_path = explanations_dir / f"scene_{scene_index:03d}_grounded.txt"
    grounded_summary_text = None

    if duration_seconds:
        try:
            grounded_summary_text = _openai_generate_grounded_summary(
                base_url=base_url,
                api_key=api_key,
                model=model,
                explanation_text=explanation,
                scene_transcript=scene_txt,
                duration_seconds=duration_seconds,
            )
            grounded_cache_path.write_text(grounded_summary_text, encoding="utf-8")
        except Exception:
            grounded_summary_text = None

    explanation_audio_url: Optional[str] = None
    if _try_write_explanation_narration(
        base_url=base_url,
        api_key=api_key,
        explanation_text=explanation,
        mp3_path=audio_path,
    ):
        explanation_audio_url = f"/api/explanations/{video_id}/{audio_name}"

    summary_audio_url: Optional[str] = None
    summary_audio_name = f"scene_{scene_index:03d}_summary.mp3"
    summary_audio_path = explanations_dir / summary_audio_name

    if summary_text and api_key:
        try:
            _openai_tts_to_mp3(
                base_url=base_url,
                api_key=api_key,
                text=summary_text,
                out_path=summary_audio_path,
            )
            summary_audio_url = f"/api/explanations/{video_id}/{summary_audio_name}"
        except Exception:
            summary_audio_url = None

    grounded_audio_url: Optional[str] = None
    grounded_audio_name = f"scene_{scene_index:03d}_grounded.mp3"
    grounded_audio_path = explanations_dir / grounded_audio_name

    if grounded_summary_text and api_key:
        try:
            _openai_tts_to_mp3(
                base_url=base_url,
                api_key=api_key,
                text=grounded_summary_text,
                out_path=grounded_audio_path,
            )
            grounded_audio_url = f"/api/explanations/{video_id}/{grounded_audio_name}"
        except Exception:
            grounded_audio_url = None

    return JSONResponse(
        {
            "explanation": explanation,
            "summary": summary_text,
            "grounded_summary": grounded_summary_text,
            "duration_seconds": duration_seconds,
            "cached": False,
            "scene_index": scene_index,
            "frame_count": len(frames),
            "frame_interval_ms": frame_interval_ms,
            "explanation_audio_url": explanation_audio_url,
            "summary_audio_url": summary_audio_url,
            "grounded_summary_audio_url": grounded_audio_url,
        },
    )


def _parse_enhanced_transcript_segments(enhanced_text: str) -> List[Dict]:
    """Parse enhanced transcript into dialogue and enhancement segments.

    Input format: [timestamp] [speaker] dialogue [enhancement1] [enhancement2]...

    Returns: List of dicts with keys: timestamp, speaker, dialogue, enhancement
    """
    import re

    segments = []
    raw_segments = re.split(r'\n\n+', enhanced_text.strip())

    for segment_text in raw_segments:
        if not segment_text.strip():
            continue

        # Extract timestamp [6.8s–9.7s]
        timestamp_match = re.match(r'\[([^\]]+)\]', segment_text)
        if not timestamp_match:
            continue
        timestamp = f"[{timestamp_match.group(1)}]"

        # Remove timestamp and extract rest
        rest = segment_text[timestamp_match.end():].strip()

        # Extract speaker [Female/Park Visitor]
        speaker_match = re.match(r'\[([^\]]+)\]', rest)
        if not speaker_match:
            continue
        speaker = f"[{speaker_match.group(1)}]"

        # Remove speaker and extract dialogue + enhancements
        dialogue_and_enhancements = rest[speaker_match.end():].strip()

        # Extract all bracketed enhancements from the end
        enhancements = []
        enhancement_pattern = r'\s*\[([^\]]+)\]\s*$'

        while True:
            enhancement_match = re.search(enhancement_pattern, dialogue_and_enhancements)
            if not enhancement_match:
                break
            enhancements.insert(0, enhancement_match.group(1))
            dialogue_and_enhancements = dialogue_and_enhancements[:enhancement_match.start()]

        dialogue = dialogue_and_enhancements.strip()
        enhancement_text = " ".join(enhancements) if enhancements else ""

        segments.append({
            "timestamp": timestamp,
            "speaker": speaker,
            "dialogue": dialogue,
            "enhancement": enhancement_text,
        })

    return segments


def _create_silence_mp3(duration_seconds: float, output_path: Path) -> None:
    """Generate silent MP3 file using ffmpeg."""
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    subprocess.run(
        [
            ffmpeg,
            "-f", "lavfi",
            "-i", f"anullsrc=r=24000:cl=mono",
            "-t", str(duration_seconds),
            "-q:a", "9",
            "-acodec", "libmp3lame",
            str(output_path),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )


def _merge_mp3_files(mp3_files: List[Path], output_path: Path) -> None:
    """Merge multiple MP3 files using ffmpeg concat demuxer."""
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    # Create concat demuxer file
    concat_file = output_path.parent / ".concat_list.txt"
    lines = [f"file '{f.absolute()}'" for f in mp3_files]
    concat_file.write_text("\n".join(lines))

    try:
        subprocess.run(
            [
                ffmpeg,
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(output_path),
            ],
            capture_output=True,
            check=True,
            timeout=60,
        )
    finally:
        concat_file.unlink(missing_ok=True)


def _openai_tts_to_mp3_dual_voice(
    *,
    base_url: str,
    api_key: str,
    segments: List[Dict],
    out_path: Path,
) -> None:
    """Generate dual-voice TTS with different voices for dialogue and enhancements."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        all_mp3_files = []

        for idx, segment in enumerate(segments):
            # Generate dialogue audio (strong voice, normal speed)
            dialogue_text = f"{segment['speaker']} {segment['dialogue']}"
            dialogue_mp3 = tmp_path / f"dialogue_{idx:03d}.mp3"

            try:
                _openai_tts_to_mp3(
                    base_url=base_url,
                    api_key=api_key,
                    text=dialogue_text,
                    out_path=dialogue_mp3,
                    voice=config.OPENAI_TTS_VOICE,
                    speed=config.OPENAI_TTS_DIALOGUE_SPEED,
                )
                all_mp3_files.append(dialogue_mp3)
            except Exception as e:
                print(f"Failed to generate dialogue audio for segment {idx}: {e}")
                continue

            # Generate enhancement audio (softer voice, slower speed) if enhancements exist
            if segment["enhancement"].strip():
                # Add 0.5s silence between dialogue and enhancement for clarity
                silence_mp3 = tmp_path / f"silence_{idx:03d}.mp3"
                try:
                    _create_silence_mp3(0.5, silence_mp3)
                    all_mp3_files.append(silence_mp3)
                except Exception as e:
                    print(f"Failed to generate silence: {e}")

                enhancement_mp3 = tmp_path / f"enhancement_{idx:03d}.mp3"

                try:
                    _openai_tts_to_mp3(
                        base_url=base_url,
                        api_key=api_key,
                        text=segment["enhancement"],
                        out_path=enhancement_mp3,
                        voice=config.OPENAI_TTS_ENHANCEMENT_VOICE,
                        speed=config.OPENAI_TTS_ENHANCEMENT_SPEED,
                    )
                    all_mp3_files.append(enhancement_mp3)
                except Exception as e:
                    print(f"Failed to generate enhancement audio for segment {idx}: {e}")

        # Merge all MP3 files
        if all_mp3_files:
            _merge_mp3_files(all_mp3_files, out_path)
        else:
            raise RuntimeError("No MP3 files generated for merging")


@app.post("/api/enhance-transcript/{video_id}/{scene_index}")
def enhance_transcript(
    video_id: str,
    scene_index: int,
    regenerate: bool = Query(False),
):
    """Enhance scene transcript with visual/contextual details.

    Takes scene-specific transcript and grounded summary, then uses a second AI model
    to add speaker identification, objects, actions, and scene context.

    Args:
        video_id: Video identifier
        scene_index: 0-indexed scene number
        regenerate: If true, regenerate even if cached

    Returns:
        JSON with enhanced_transcript text and audio URL
    """
    if not config.OPENAI_API_KEY:
        return JSONResponse(
            {"error": "OPENAI_API_KEY not configured"},
            status_code=503,
        )

    video_dir = OUTPUTS / video_id
    if not video_dir.is_dir():
        raise HTTPException(status_code=404, detail="Video not found")

    # Determine which provider was used
    providers = ["whisper", "assemblyai"]
    provider = None
    for p in providers:
        if (video_dir / "transcripts" / f"scene_{scene_index:03d}_{p}.json").is_file():
            provider = p
            break

    if not provider:
        raise HTTPException(status_code=404, detail="Scene transcript not found")

    # Check cache
    enhanced_cache = video_dir / "explanations" / f"scene_{scene_index:03d}_enhanced_transcript.txt"
    if enhanced_cache.is_file() and not regenerate:
        cached_text = enhanced_cache.read_text(encoding="utf-8").strip()
        audio_url = f"/api/explanations/{video_id}/scene_{scene_index:03d}_enhanced_transcript.mp3"
        return JSONResponse(
            {
                "enhanced_transcript": cached_text,
                "enhanced_transcript_audio_url": audio_url,
                "cached": True,
            }
        )

    # Load scene transcript
    scene_transcript_path = video_dir / "transcripts" / f"scene_{scene_index:03d}_{provider}.json"
    try:
        scene_data = json.loads(scene_transcript_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load scene transcript: {e}")

    # Build scene transcript text
    scene_transcript_text = _format_transcript_segments(scene_data)

    # Load grounded summary
    grounded_path = video_dir / "explanations" / f"scene_{scene_index:03d}_grounded.txt"
    if not grounded_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="Grounded summary not generated. Generate explanation first.",
        )
    grounded_summary_text = grounded_path.read_text(encoding="utf-8").strip()

    # Load video context if available
    video_context_text = ""
    context_path = video_dir / "video_context.json"
    if context_path.is_file():
        try:
            video_context = load_video_context(context_path)
            video_context_text = format_video_context_for_prompt(
                video_context,
                character_confidence_threshold=config.CHARACTER_CONFIDENCE_THRESHOLD_PERCENT,
            )
        except Exception:
            pass

    # Generate enhancement
    try:
        enhanced_text = _openai_enhance_scene_transcript(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_EXPLAIN_MODEL,
            scene_transcript=scene_transcript_text,
            grounded_summary=grounded_summary_text,
            video_context_text=video_context_text,
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"Enhancement failed: {e}"},
            status_code=500,
        )

    # Cache enhanced transcript
    video_dir / "explanations"
    (video_dir / "explanations").mkdir(parents=True, exist_ok=True)
    enhanced_cache.write_text(enhanced_text, encoding="utf-8")

    # Generate dual-voice TTS audio
    audio_url = None
    mp3_path = video_dir / "explanations" / f"scene_{scene_index:03d}_enhanced_transcript.mp3"
    if config.OPENAI_API_KEY and not mp3_path.is_file():
        try:
            # Parse enhanced transcript into dialogue and enhancement segments
            segments = _parse_enhanced_transcript_segments(enhanced_text)

            # Generate dual-voice audio
            if segments:
                _openai_tts_to_mp3_dual_voice(
                    base_url=config.OPENAI_BASE_URL,
                    api_key=config.OPENAI_API_KEY,
                    segments=segments,
                    out_path=mp3_path,
                )
            else:
                # Fallback to single-voice if parsing fails
                text_to_speak = _tts_input_clip(enhanced_text)
                _openai_tts_to_mp3(
                    base_url=config.OPENAI_BASE_URL,
                    api_key=config.OPENAI_API_KEY,
                    text=text_to_speak,
                    out_path=mp3_path,
                )
        except Exception as e:
            print(f"TTS generation failed for enhanced transcript: {e}")
            # Attempt single-voice fallback
            try:
                text_to_speak = _tts_input_clip(enhanced_text)
                _openai_tts_to_mp3(
                    base_url=config.OPENAI_BASE_URL,
                    api_key=config.OPENAI_API_KEY,
                    text=text_to_speak,
                    out_path=mp3_path,
                )
            except Exception as fallback_e:
                print(f"Fallback single-voice TTS also failed: {fallback_e}")

    if mp3_path.is_file():
        audio_url = f"/api/explanations/{video_id}/scene_{scene_index:03d}_enhanced_transcript.mp3"

    return JSONResponse(
        {
            "enhanced_transcript": enhanced_text,
            "enhanced_transcript_audio_url": audio_url,
            "cached": False,
        }
    )


@app.get("/api/explanations/{video_id}/{filename}")
def get_explanation_asset(video_id: str, filename: str):
    safe = Path(filename).name
    if not safe.startswith("scene_") or not safe.endswith(".mp3"):
        raise HTTPException(status_code=404)
    path = OUTPUTS / video_id / "explanations" / safe
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(
        str(path),
        media_type="audio/mpeg",
        filename=safe,
    )


@app.get("/api/clips/{video_id}/{kind}/{filename}")
def get_clip(video_id: str, kind: str, filename: str):
    if kind not in ("shots", "scenes", "scene_audio"):
        raise HTTPException(status_code=404)
    safe = Path(filename).name
    path = OUTPUTS / video_id / kind / safe
    if not path.is_file():
        raise HTTPException(status_code=404)
    media = (
        "audio/mp4"
        if kind == "scene_audio"
        else "video/mp4"
    )
    return FileResponse(
        str(path),
        media_type=media,
        filename=safe,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(ROOT),
    )
