.PHONY: help setup install run dev clean check-ffmpeg

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "test-clips - Video Analysis Web App"
	@echo ""
	@echo "Usage:"
	@echo "  make setup          Set up virtual environment and install dependencies"
	@echo "  make install        Install/update dependencies"
	@echo "  make run            Run the application"
	@echo "  make dev            Setup and run (all-in-one)"
	@echo "  make check-ffmpeg   Verify FFmpeg and ffprobe are installed"
	@echo "  make clean          Remove virtual environment and cached files"
	@echo ""

check-ffmpeg:
	@command -v ffmpeg >/dev/null 2>&1 || { echo "❌ ffmpeg not found. Install with: brew install ffmpeg"; exit 1; }
	@command -v ffprobe >/dev/null 2>&1 || { echo "❌ ffprobe not found. Install with: brew install ffmpeg"; exit 1; }
	@echo "✅ ffmpeg and ffprobe found"

$(VENV):
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "✅ Virtual environment created"

setup: check-ffmpeg $(VENV) install
	@echo "✅ Setup complete! Run 'make run' to start the app"

install: $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

run: $(VENV) check-ffmpeg
	@echo "Starting test-clips application..."
	@echo "App will be available at http://127.0.0.1:8000"
	$(PYTHON) -m app.main

dev: setup run

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Clean complete"
