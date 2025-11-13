# CustomerAssist - Customer Service AI Assistant

This package provides a **FastAPI** demo that exposes a collection of AI functionalities (NLP, Computer Vision, and Audio/ASR) via HTTP endpoints, primarily using **Hugging Face transformers pipelines**. It is designed to be the backend for an AI-powered customer service platform.

---

## 🚀 Quickstart (Local Development)

1.  **Create Virtual Environment & Install Dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
    pip install -r requirements.txt
    ```

2.  **Run the FastAPI Server:** (Example port 8010)
    ```bash
    uvicorn app.main:app --reload --port 8010
    ```

3.  **Access Interactive API Docs (Swagger UI):**
    ```
    [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs)
    ```

---

## 📦 Project Architecture Overview

The codebase is organized into modular components centered around three categories of AI models, utilizing a declarative configuration approach.

| File/Directory | Description |
| :--- | :--- |
| `app/main.py` | FastAPI application, HTTP endpoints, **Prometheus metrics**, and integration of LLMOps utilities. |
| `app/models.py` | Centralized **Hugging Face pipeline** initialization, helper functions (e.g., token counting, defect detection logic), grouped by AI domain (NLP, CV, ASR). |
| `app/config.py` | Configuration for all model IDs (from Hugging Face), prompt templates, and device mapping (`QA_MODEL`, `SUMMARIZER`, `ASR_MODEL`, etc.). |
| `app/monitor.py` | LLMOps module for logging and aggregating operational/quality metrics (e.g., latency, error rate, quality score). |
| `app/utils.py` | Utility functions for **Rate Limiting** (Token Bucket), **Caching** (TTL), and **Resilience** (Retry with Backoff). |
| `requirements.txt` | Python dependencies (FastAPI, uvicorn, transformers, torch, opencv, easyocr). |

---

## 🌐 Main API Endpoints

The API is structured around core customer service needs:

| Endpoint | Method | Functionality | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| `/` | `GET` | Health/Status check. | N/A | `{"message": ...}` |
| `/question-answering` | `POST` | Extract concise answers from a long context text. | `{"context": "...", "question": "..."}` | `{"question": "...", "answer": "..."}` |
| `/summarize-text` | `POST` | Summarize conversations or long documents, focusing on issue/resolution. | `{"text": "..."}` | `{"summary": "..."}` |
| `/translate-text` | `POST` | Translate text between specified languages. | `{"text": "...", "src_lang": "hi", "target_lang": "eng"}` | `{"translation": "..."}` |
| `/check-item-return-eligibility` | `POST` | **Computer Vision** defect detection to determine return eligibility. | Upload `file` (image) | `{"detection": {...}, "eligible_for_return": true/false}` |
| `/audio-transcribe` | `POST` | **ASR** (Speech Recognition) to convert audio to text. | Upload `file` (audio: wav/mp3) | `{"transcription": "..."}` |
| `/audio-transcribe-translate` | `POST` | **ASR + Translation** chain for cross-lingual audio support. | Upload `file` (audio: wav/mp3) | `{"translation": "..."}` |
| `/metrics` | `GET` | **Prometheus** metrics export (latency, call counts, errors). | N/A | Prometheus Text Format |
| `/llmops/summary` | `GET` | Aggregated LLMOps metrics (latency P50/P95, error rate, token usage, quality score). | N/A | JSON |

---

## ⚙️ LLMOps and Resilience Highlights

The application incorporates key operational monitoring and resilience features:

### Monitoring & Observability
* **Structured Logging:** All LLM interactions are logged in a structured format (JSON) via `app/monitor.py` for downstream analysis.
* **Operational Metrics:** The `/llmops/summary` endpoint aggregates metrics from the logs, including:
    * **Latency:** Average, P50, and P95 latency (in ms).
    * **Throughput:** Requests per second.
    * **Quality Score:** An automated score (0.0 to 1.0) is assigned to each response based on output length and error presence, providing an immediate proxy for quality.
    * **Token Usage:** Tracking total tokens per request for cost analysis.
* **Prometheus:** Standardized metrics (Counters for calls/errors, Histograms for latency) are exposed on `/metrics`.

### Resilience and Cost Control
* **Caching:** Responses for `question-answering`, `summarize-text`, and `translate-text` are cached (via `app/utils.RESPONSE_CACHE`) for 5 minutes (`ttl_seconds=300`) to improve performance and reduce redundant LLM calls.
* **Rate Limiting:** A Token Bucket algorithm (`app/utils.GLOBAL_RATE_LIMITER`) is used to enforce a rate limit of **10 requests max burst** (refilling at **5 tokens per 60 seconds**) across all endpoints.
* **Retries with Backoff:** Key model functions (`answer_question`, `summarize_text`, `translate_text`, `transcribe_audio`) use the `@retry_with_backoff` decorator (max 3 retries with jittered exponential backoff) to handle transient failures.
* **Input/Output Limits:** Constraints are enforced in `app/main.py` (e.g., `MAX_TEXT_INPUT_CHAR_COUNT=4000`, `MAX_IMAGE_FILE_SIZE=5MB`, `MAX_RESPONSE_LENGTH=1500`) to prevent abuse and manage costs.

### Configuration Management
* **Model Registry (`app/config.py`):** Model IDs are declared as constants (e.g., `QA_MODEL`, `SUMMARIZER`) to allow changing models without altering business logic.
* **Device Handling:** The application checks for MPS (macOS GPU) support but defaults pipelines to CPU (`DEVICE_ID = -1`) for broader `transformers` compatibility.