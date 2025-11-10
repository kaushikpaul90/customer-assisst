# CustomerAssist - Customer Service AI Assistant (Minimal Distribution)

This package contains a minimal API demonstrating an customer service focused AI assistant.

## What's included
- FastAPI server with endpoints: /qa, /summarize, /explain, /upload-image
- Lightweight model wrappers using Hugging Face pipelines
- Fine-tune and evaluation scripts for QA (SQuAD subset)

## Getting started (local)
1. Create venv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
   # CustomerAssist - Customer Service AI Assistant (Minimal Distribution)

   This repository provides a small FastAPI demo that exposes NLP, Computer Vision
   and Audio (ASR) functionality via HTTP endpoints. The code centralizes
   Hugging Face pipeline initialization in `app/models.py` and configuration in
   `app/config.py`.

   ## What changed (recent refactor)
   - Endpoints and handler names were renamed to clearer REST-style paths.
   - Pipelines and model identifiers were consolidated into `app/config.py`.
   - The codebase is organized into three categories: NLP, Computer Vision, and
      Audio/ASR (see `app/models.py`).
   - A small unused helper (`explain_defect`) was removed to keep the codebase
      tidy.

   ## Quickstart (local)
   1. Create virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   2. Run the server (example port 8010 used in development):

   ```bash
   uvicorn app.main:app --reload --port 8010
   ```

   3. Open the interactive API docs at:

      http://127.0.0.1:8010/docs

   ## Main endpoints

   - POST /question-answering
      - Request: {"context": "long text...", "question": "..."}
      - Returns: {"question": ..., "answer": ...}

   - POST /summarize-text
      - Request: {"text": "conversation or long text"}
      - Returns: {"summary": "..."}

   - POST /translate-text
      - Request: {"text": "...", "src_lang": "hi|eng|...", "target_lang": "hi|eng|..."}
      - Note: short codes are mapped: `hi` -> `hin_Deva`, `eng` -> `eng_Latn` before
         calling the translation pipeline.
      - Returns: {"translation": "..."}

   - POST /image/check-return-eligibility
      - Upload: form field `file` (image)
      - Runs defect detection and returns `{ "detection": {...}, "eligibility": "eligible|not_eligible" }`

   - POST /audio/transcribe
      - Upload audio file (wav/mp3) as `file` form field
      - Returns: {"transcription": "..."}

   - POST /audio/transcribe-translate
      - Upload audio, returns translated text (calls ASR + translation)

   - GET /
      - Health / status message

   ## Configuration highlights (`app/config.py`)

   - All model IDs are declared as constants (e.g. `QA_MODEL`, `SUMMARIZER`,
      `IMG_CLASSIFIER`, `DEFECT_DETECTOR`, `ASR_MODEL`) so you can change models
      without editing business logic.
   - Device handling on macOS: the repo detects MPS via `torch.backends.mps.is_available()`
      and sets `DEVICE = "mps"` when available. However, because `transformers`
      pipelines do not yet fully support MPS, most pipelines are configured to use
      the CPU device by default (see `DEVICE_ID`/`*_PIPELINE_KWARGS`).

   ## Project layout

   - `app/main.py` — FastAPI endpoints (thin wrappers)
   - `app/models.py` — pipeline initialization + helper functions grouped into:
      - NLP PIPELINES & FUNCTIONS
      - COMPUTER VISION PIPELINES & FUNCTIONS
      - AUDIO & SPEECH RECOGNITION PIPELINES & FUNCTIONS
   - `app/config.py` — semantic model/task constants and pipeline kwargs
   - `app/utils.py` — small utilities (metrics/time)

   ## Examples (curl)

   Question answering:
   ```bash
   curl -s -X POST "http://127.0.0.1:8010/question-answering" \
      -H "Content-Type: application/json" \
      -d '{"context":"Long document text...", "question":"What is the issue?"}'
   ```

   Check item return eligibility (image):
   ```bash
   curl -s -X POST "http://127.0.0.1:8010/image/check-return-eligibility" \
      -F "file=@/path/to/photo.jpg"
   ```

   Transcribe audio:
   ```bash
   curl -s -X POST "http://127.0.0.1:8010/audio/transcribe" -F "file=@/path/to/file.wav"
   ```

   ## Notes and next steps

   - If you depend on older endpoints (e.g. `/qa`, `/summarize`, `/scan-item`,
      `/asr`), update clients or ask for a compatibility wrapper that routes old
      paths to the new ones.
   - For production, consider converting pipeline initialization into lazy
      factories to reduce import-time overhead and speed up cold starts.

   If you'd like, I can add backward-compatible proxy routes, unit tests for
   the new endpoints, or example Postman collections.
