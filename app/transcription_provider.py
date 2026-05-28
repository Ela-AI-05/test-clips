"""Transcription provider abstraction layer supporting multiple backends (Whisper, AssemblyAI)."""

import json
import os
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

from app import config


@dataclass
class TranscriptionResult:
    """Unified transcription result format across all providers."""

    text: str
    segments: List[Dict]
    language: Optional[str] = None
    language_probability: Optional[float] = None
    diarization: Dict = field(default_factory=lambda: {"enabled": False, "speaker_count": 0})
    provider: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class TranscriptionProvider(ABC):
    """Abstract base class for transcription providers."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio file and return structured result.

        Args:
            audio_path: Path to audio file (M4A format)

        Returns:
            TranscriptionResult with segments, speaker info, timestamps

        Raises:
            RuntimeError: If transcription fails
        """
        pass


class WhisperTranscriber(TranscriptionProvider):
    """Transcription provider using faster-whisper."""

    def __init__(self):
        self._model = None
        self._model_name: Optional[str] = None

    def _get_whisper_model(self):
        """Lazy load and cache Whisper model."""
        model_name = config.WHISPER_MODEL
        if self._model is not None and self._model_name == model_name:
            return self._model
        from faster_whisper import WhisperModel

        device = config.WHISPER_DEVICE
        compute = (
            config.WHISPER_COMPUTE_TYPE_GPU
            if device == "cuda"
            else config.WHISPER_COMPUTE_TYPE_CPU
        )
        self._model = WhisperModel(model_name, device=device, compute_type=compute)
        self._model_name = model_name
        return self._model

    def _audio_to_wav16k_mono(self, ffmpeg: str, src: Path, dst: Path) -> None:
        """Convert audio to 16kHz mono WAV for diarization."""
        import subprocess

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

    def _human_speaker_labels(self, turns: List[Tuple[float, float, str]]) -> Dict[str, str]:
        """Convert raw speaker labels to human-readable format."""
        ordered: List[str] = []
        seen: Set[str] = set()
        for _a, _b, raw in turns:
            if raw not in seen:
                seen.add(raw)
                ordered.append(raw)
        return {label: f"Speaker {i + 1}" for i, label in enumerate(ordered)}

    def _best_speaker_for_segment(
        self,
        seg_start: float,
        seg_end: float,
        turns: List[Tuple[float, float, str]],
        label_map: Dict[str, str],
    ) -> Optional[str]:
        """Find best speaker for segment based on overlap with diarization turns."""
        best_raw: Optional[str] = None
        best_ov = 0.0
        for t0, t1, raw in turns:
            overlap = max(0.0, min(seg_end, t1) - max(seg_start, t0))
            if overlap > best_ov:
                best_ov = overlap
                best_raw = raw
        if best_raw is None or best_ov <= 0:
            return None
        return label_map.get(best_raw)

    def _diarize_speakers(self, wav_path: Path) -> Optional[List[Tuple[float, float, str]]]:
        """Run speaker diarization using pyannote if available."""
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
        rows: List[Tuple[float, float, str]] = []
        for turn, _track, speaker in diarization.itertracks(yield_label=True):
            rows.append((float(turn.start), float(turn.end), str(speaker)))
        return rows if rows else None

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio using Whisper with optional speaker diarization."""
        import shutil

        turns: Optional[List[Tuple[float, float, str]]] = None
        label_map: Dict[str, str] = {}
        diar_meta: dict = {
            "enabled": False,
            "speaker_count": 0,
            "note": (
                "Set HF_TOKEN (Hugging Face) and accept the model terms for "
                "pyannote/speaker-diarization-3.1; install pyannote.audio. "
                "Otherwise only plain transcription is returned."
            ),
        }

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            tmp_wav = Path(tempfile.gettempdir()) / (
                f"test-clips-diar-{uuid.uuid4().hex}.wav"
            )
            try:
                self._audio_to_wav16k_mono(ffmpeg, audio_path, tmp_wav)
                turns = self._diarize_speakers(tmp_wav)
                if turns:
                    label_map = self._human_speaker_labels(turns)
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

        model = self._get_whisper_model()
        segments_iter, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
        )
        segments_out: List[Dict] = []
        full_parts: List[str] = []
        for seg in segments_iter:
            words_out: List[Dict] = []
            speaker_label: Optional[str] = None
            if turns:
                speaker_label = self._best_speaker_for_segment(
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

        return TranscriptionResult(
            text=" ".join(full_parts).strip(),
            segments=segments_out,
            language=info.language,
            language_probability=round(float(info.language_probability), 4),
            diarization=diar_meta,
            provider="whisper",
        )


class AssemblyAITranscriber(TranscriptionProvider):
    """Transcription provider using AssemblyAI API."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("AssemblyAI API key is required")
        self.api_key = api_key

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio using AssemblyAI with speaker diarization."""
        from assemblyai import Client, Settings, TranscriptionConfig, Transcriber

        try:
            settings = Settings(api_key=self.api_key)
            client = Client(settings=settings)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AssemblyAI client: {e}") from e

        try:
            config = TranscriptionConfig(
                speaker_labels=True,
                speech_models=["universal-3-pro", "universal-2"],
            )
            transcriber_obj = Transcriber(client=client)
            transcript = transcriber_obj.transcribe(str(audio_path), config=config)
        except Exception as e:
            raise RuntimeError(f"AssemblyAI transcription failed: {e}") from e

        if not transcript or transcript.status.value != "completed":
            raise RuntimeError(
                f"AssemblyAI transcription did not complete: {transcript.status}"
            )

        segments_out: List[Dict] = []
        full_parts: List[str] = []
        speaker_set: Set[Optional[int]] = set()

        if transcript.utterances:
            for utterance in transcript.utterances:
                start = utterance.start / 1000.0  # ms to seconds
                end = utterance.end / 1000.0
                text = (utterance.text or "").strip()
                speaker_id = utterance.speaker

                if text:
                    full_parts.append(text)
                    speaker_set.add(speaker_id)

                    speaker_label = None
                    if speaker_id is not None:
                        speaker_label = f"Speaker {speaker_id}"

                    row: dict = {
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": text,
                        "words": [],
                    }
                    if speaker_label:
                        row["speaker"] = speaker_label

                    segments_out.append(row)

        diar_meta = {
            "enabled": len(speaker_set) > 1 or (len(speaker_set) == 1 and None not in speaker_set),
            "speaker_count": max(len(speaker_set) - (1 if None in speaker_set else 0), 0),
            "note": "Speaker diarization enabled" if speaker_set else "No speakers detected",
        }

        return TranscriptionResult(
            text=" ".join(full_parts).strip(),
            segments=segments_out,
            language=transcript.language_code if hasattr(transcript, 'language_code') else "unknown",
            language_probability=None,
            diarization=diar_meta,
            provider="assemblyai",
        )


def get_transcriber(provider: Optional[str] = None) -> TranscriptionProvider:
    """Factory function to get appropriate transcriber based on configuration.

    Args:
        provider: Override provider name ("whisper" or "assemblyai"). Uses config default if None.

    Returns:
        Appropriate TranscriptionProvider instance

    Raises:
        ValueError: If provider is invalid or required config/API key is missing
    """
    selected_provider = (provider or config.TRANSCRIPTION_PROVIDER).lower()

    if selected_provider == "whisper":
        return WhisperTranscriber()
    elif selected_provider == "assemblyai":
        api_key = config.ASSEMBLYAI_API_KEY
        if not api_key:
            raise ValueError(
                "ASSEMBLYAI_API_KEY environment variable is required for AssemblyAI provider"
            )
        return AssemblyAITranscriber(api_key)
    else:
        raise ValueError(
            f"Unknown transcription provider: {selected_provider}. Options: whisper, assemblyai"
        )
