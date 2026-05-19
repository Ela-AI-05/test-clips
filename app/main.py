import base64
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

import requests


SCENE_EXPLAIN_SYSTEM_PROMPT = """You are a film and media analyst helping viewers understand a scene.

Work in two clear mental steps:
1) First, read the **full-film transcript** (all scenes’ dialogue together). Build a loose sense of ongoing story, tone, and recurring themes—only as far as the text supports.
2) Then focus on **this scene only**, using:
   - The **scene-specific transcript** (dialogue and speakers if tagged),
   - The **video frames** provided in order (each from a moment in the scene). From the frames, infer **setting, action, blocking, cinematography**, and **facial/body language** to reason about **mood and emotion** (e.g. tension, joy, fear). Be honest when unclear.

Your task: write a clear **scene explanation** for the user—what is happening, why it matters in context, and how it feels emotionally. When you infer emotion from visuals, say what you see (expressions, posture, staging) that supports it. Do not invent plot facts that contradict the transcripts; do not name actors unless they appear in the text."""

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
STATIC = ROOT / "static"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

_whisper_model = None
_whisper_model_name: str | None = None


def _get_whisper_model():  # pragma: no cover — loads ML weights on first use
    global _whisper_model, _whisper_model_name
    model_name = os.environ.get("WHISPER_MODEL", "base")
    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model
    from faster_whisper import WhisperModel

    device = os.environ.get("WHISPER_DEVICE", "cpu")
    compute = (
        os.environ.get("WHISPER_COMPUTE", "float16")
        if device == "cuda"
        else os.environ.get("WHISPER_COMPUTE", "int8")
    )
    _whisper_model = WhisperModel(model_name, device=device, compute_type=compute)
    _whisper_model_name = model_name
    return _whisper_model


def _which_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _audio_to_wav16k_mono(ffmpeg: str, src: Path, dst: Path) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        check=True,
    )


def _human_speaker_labels(
    turns: list[tuple[float, float, str]],
) -> dict[str, str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for _a, _b, raw in turns:
        if raw not in seen:
            seen.add(raw)
            ordered.append(raw)
    return {label: f"Speaker {i + 1}" for i, label in enumerate(ordered)}


def _best_speaker_for_segment(
    seg_start: float,
    seg_end: float,
    turns: list[tuple[float, float, str]],
    label_map: dict[str, str],
) -> str | None:
    best_raw: str | None = None
    best_ov = 0.0
    for t0, t1, raw in turns:
        overlap = max(0.0, min(seg_end, t1) - max(seg_start, t0))
        if overlap > best_ov:
            best_ov = overlap
            best_raw = raw
    if best_raw is None or best_ov <= 0:
        return None
    return label_map.get(best_raw)


def _diarize_speakers(wav_path: Path) -> list[tuple[float, float, str]] | None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    ).strip()
    if not token:
        return None
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return None
    try:
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token,
            )
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
    except Exception:
        return None
    try:
        diarization = pipeline(str(wav_path))
    except Exception:
        return None
    rows: list[tuple[float, float, str]] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        rows.append((float(turn.start), float(turn.end), str(speaker)))
    return rows if rows else None


def _transcribe_audio_file(
    audio_path: Path,
    *,
    ffmpeg: str | None,
) -> dict:
    turns: list[tuple[float, float, str]] | None = None
    label_map: dict[str, str] = {}
    diar_meta: dict = {
        "enabled": False,
        "speaker_count": 0,
        "note": (
            "Set HF_TOKEN (Hugging Face) and accept the model terms for "
            "pyannote/speaker-diarization-3.1; install pyannote.audio. "
            "Otherwise only plain transcription is returned."
        ),
    }

    if ffmpeg:
        tmp_wav = Path(tempfile.gettempdir()) / (
            f"test-clips-diar-{uuid.uuid4().hex}.wav"
        )
        try:
            _audio_to_wav16k_mono(ffmpeg, audio_path, tmp_wav)
            turns = _diarize_speakers(tmp_wav)
            if turns:
                label_map = _human_speaker_labels(turns)
                diar_meta = {
                    "enabled": True,
                    "speaker_count": len(label_map),
                }
        except Exception:
            diar_meta = {
                "enabled": False,
                "speaker_count": 0,
                "note": "Diarization could not run (see server logs).",
            }
        finally:
            tmp_wav.unlink(missing_ok=True)

    model = _get_whisper_model()
    segments_iter, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
    )
    segments_out: list[dict] = []
    full_parts: list[str] = []
    for seg in segments_iter:
        words_out: list[dict] = []
        speaker_label: str | None = None
        if turns:
            speaker_label = _best_speaker_for_segment(
                seg.start,
                seg.end,
                turns,
                label_map,
            )
        if seg.words:
            for w in seg.words:
                wd = {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.word.strip(),
                }
                if speaker_label:
                    wd["speaker"] = speaker_label
                words_out.append(wd)
        seg_text = (seg.text or "").strip()
        full_parts.append(seg_text)
        row: dict = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg_text,
            "words": words_out,
        }
        if speaker_label:
            row["speaker"] = speaker_label
        segments_out.append(row)
    return {
        "language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "diarization": diar_meta,
        "segments": segments_out,
        "text": " ".join(full_parts).strip(),
    }


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
    manager.add_detector(ContentDetector())
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
    cmd: list[str] = [
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
    shots: list[tuple[float, float]],
    shots_per_scene: int,
) -> list[list[tuple[float, float]]]:
    if shots_per_scene < 1:
        raise ValueError("shots_per_scene must be >= 1")
    return [shots[i : i + shots_per_scene] for i in range(0, len(shots), shots_per_scene)]


def _concat_shot_files(
    ffmpeg: str,
    shot_paths: list[Path],
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


def _transcript_paths_sorted(transcripts_dir: Path) -> list[Path]:
    files = list(transcripts_dir.glob("scene_*.json"))

    def sort_key(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    return sorted(files, key=sort_key)


def _build_full_transcript_context(transcripts_dir: Path, max_chars: int) -> str:
    parts: list[str] = []
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


def _scene_transcript_block(transcript_path: Path | None) -> str:
    if transcript_path is None or not transcript_path.is_file():
        return "(No transcript available for this scene.)"
    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "(Transcript file could not be read.)"
    segs = data.get("segments") or []
    lines: list[str] = []
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
    return (data.get("text") or "").strip() or "(Empty transcript.)"


def _extract_scene_frames_jpeg(
    ffmpeg: str,
    video_path: Path,
    out_dir: Path,
    *,
    interval_ms: int,
    max_frames: int,
) -> list[tuple[float, Path]]:
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
    out: list[tuple[float, Path]] = []
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
    user_content: list[dict],
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "max_tokens": int(os.environ.get("OPENAI_EXPLAIN_MAX_TOKENS", "1200")),
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
        timeout=int(os.environ.get("OPENAI_EXPLAIN_TIMEOUT", "120")),
    )
    if not r.ok:
        raise RuntimeError(r.text)
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(str(data))
    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _tts_input_clip(text: str) -> str:
    limit = int(os.environ.get("OPENAI_TTS_MAX_INPUT_CHARS", "4096"))
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
) -> None:
    model = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
    voice = os.environ.get("OPENAI_TTS_VOICE", "alloy")
    url = f"{base_url.rstrip('/')}/audio/speech"
    payload = {
        "model": model,
        "voice": voice,
        "input": _tts_input_clip(text),
        "response_format": "mp3",
    }
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=int(os.environ.get("OPENAI_TTS_TIMEOUT", "120")),
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
    shots_per_scene: int = Query(default=3, ge=1, le=50),
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

    shot_entries: list[dict] = []
    shot_paths_ordered: list[Path] = []

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
    scene_entries: list[dict] = []
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
            transcript_name = f"scene_{si:03d}.json"
            transcript_path = transcripts_dir / transcript_name
            try:
                payload = _transcribe_audio_file(out_audio, ffmpeg=ffmpeg)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcription failed for scene {si}: {e}. "
                    "Ensure faster-whisper is installed and WHISPER_MODEL fits your machine.",
                ) from e
            payload["scene_index"] = si
            payload["audio_filename"] = audio_name
            payload["timeline_in_source_video"] = {
                "start": round(start_s, 3),
                "end": round(end_s, 3),
            }
            transcript_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            entry["transcript_filename"] = transcript_name
            entry["transcript_url"] = f"/api/transcript/{video_id}/{transcript_name}"

        scene_entries.append(entry)

    return {
        "video_id": video_id,
        "shots_per_scene": shots_per_scene,
        "shots": shot_entries,
        "scenes": scene_entries,
    }


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


@app.post("/api/explain-scene/{video_id}/{scene_index}")
def explain_scene(
    video_id: str,
    scene_index: int,
    frame_interval_ms: int = Query(default=500, ge=200, le=4000),
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

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if cache_path.is_file() and not regenerate:
        explanation_text = cache_path.read_text(encoding="utf-8")
        audio_url: str | None = None
        if api_key and not audio_path.is_file():
            _try_write_explanation_narration(
                base_url=base_url,
                api_key=api_key,
                explanation_text=explanation_text,
                mp3_path=audio_path,
            )
        if audio_path.is_file():
            audio_url = f"/api/explanations/{video_id}/{audio_name}"
        return JSONResponse(
            {
                "explanation": explanation_text,
                "cached": True,
                "scene_index": scene_index,
                "explanation_audio_url": audio_url,
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

    model = os.environ.get("OPENAI_EXPLAIN_MODEL", "gpt-4o-mini")
    max_frames = int(os.environ.get("EXPLAIN_MAX_FRAMES", "18"))
    max_context = int(os.environ.get("EXPLAIN_FULL_TRANSCRIPT_MAX_CHARS", "16000"))

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

        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "## Full video transcript (all scenes — context)\n\n"
                    f"{full_text}\n\n"
                    "## This scene — transcript (focus)\n\n"
                    f"{scene_txt}\n\n"
                    "## Video frames\n\n"
                    f"There are {len(frames)} still images, sampled about every "
                    f"{frame_interval_ms} ms through this scene, in order. Use them for "
                    "setting, action, faces, body language, and emotional tone."
                ),
            },
        ]
        img_detail = os.environ.get("OPENAI_IMAGE_DETAIL", "low")
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

    audio_url: str | None = None
    if _try_write_explanation_narration(
        base_url=base_url,
        api_key=api_key,
        explanation_text=explanation,
        mp3_path=audio_path,
    ):
        audio_url = f"/api/explanations/{video_id}/{audio_name}"

    return JSONResponse(
        {
            "explanation": explanation,
            "cached": False,
            "scene_index": scene_index,
            "frame_count": len(frames),
            "frame_interval_ms": frame_interval_ms,
            "explanation_audio_url": audio_url,
        },
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
