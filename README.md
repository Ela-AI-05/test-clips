# test-clips

Web app that splits a movie into **shots** and **scenes**, transcribes dialogue (Whisper or AssemblyAI) with **speaker diarization**, optionally tags speakers, and can **explain** each scene with OpenAI vision + TTS narration.

## Requirements

- **Python** 3.10+ (3.12–3.13 recommended; 3.14 may work depending on PyTorch wheels)
- [**FFmpeg**](https://ffmpeg.org/) and **ffprobe** on your `PATH` (e.g. `brew install ffmpeg` on macOS)

## Install

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run may download **Whisper** weights and (if you use diarization) **pyannote** / Hugging Face models—keep network access available.

## Configuration (optional)

Create a **`.env`** file in the project root (same folder as `README.md`). The app loads it automatically.

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | **Explain scene** (vision) and **narrated explanation** (TTS). Required for those features. |
| `OPENAI_BASE_URL` | Default `https://api.openai.com/v1` (compatible proxies OK). |
| `OPENAI_EXPLAIN_MODEL` | Default `gpt-4o-mini`. |
| `OPENAI_TTS_MODEL` | Default `tts-1`. |
| `OPENAI_TTS_VOICE` | Default `alloy`. |
| `TRANSCRIPTION_PROVIDER` | `whisper` (default) or `assemblyai`. Controls which transcription engine to use. |
| `ASSEMBLYAI_API_KEY` | Required for AssemblyAI transcription provider (get from https://assemblyai.com). |
| `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` | Optional **speaker diarization with Whisper**; accept terms for `pyannote/speaker-diarization-3.1` on Hugging Face. |
| `WHISPER_MODEL` | Default `base` (e.g. `tiny`, `small`). Only used if `TRANSCRIPTION_PROVIDER=whisper`. |

Example `.env`:

```env
OPENAI_API_KEY=sk-...
# optional:
# HF_TOKEN=hf_...
```

Do not commit `.env` (it is gitignored).

## Start the server

```bash
source .venv/bin/activate   # if not already active
python -m app.main
```

Then open **http://127.0.0.1:8000** in your browser (serve the app from this URL; do not open `index.html` as a `file://` page).

To stop the server: **Ctrl+C** in the terminal.

## Usage (short)

1. Choose a video file and select a transcription provider (Whisper or AssemblyAI).
2. Optionally check **Show shots** to display individual shot clips.
3. Click **Process** to analyze the video (shot detection, scene grouping, full-video transcription).
4. Review **Shots** (if enabled) and **Scenes**; click **Transcript** to view full transcription with speaker labels and timestamps.
5. Click **View Full Transcript** to see the complete video transcription with all speakers and timings.
6. Click **Explain scene** on any scene for AI-powered analysis (requires `OPENAI_API_KEY`).

Outputs are stored under `outputs/<session-id>/` (gitignored).

## Troubleshooting

- **Process / API errors**: Confirm FFmpeg works (`ffmpeg -version`) and the tab is loaded from `http://127.0.0.1:8000`.
- **Explain scene fails**: Check `OPENAI_API_KEY` in `.env` and restart the server after editing `.env`.
- **Heavy installs**: `torch` and `pyannote.audio` are large; use a fresh venv if installs conflict.
