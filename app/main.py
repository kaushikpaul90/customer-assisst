"""
REST API surface for Customer Assist.

This module defines the FastAPI application and the public endpoints
used by the demo. Endpoints are thin wrappers that call into
`app.models` and record simple latency metrics using `app.utils`.

Keep this file light: avoid heavy imports or model initialization here
so FastAPI's import time remains predictable. Heavy model code
belongs in `app.models` where it can be refactored into lazy factories.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.models import (
    answer_question, detect_defect, summarize_text, transcribe_and_translate_audio, transcribe_audio, translate_text
)
from app.utils import timeit, record_metric

# Try to include llmops router if available (non-intrusive).
# If llmops is not present, app still works exactly the same.
try:
    from app.llmops import create_router as _create_llmops_router
    _llmops_router = _create_llmops_router()
except Exception:
    _llmops_router = None

app = FastAPI(title="Customer Assist AI", description="AI-powered Customer Service Assistant", version="1.0")

# mount llmops router if available
if _llmops_router is not None:
    app.include_router(_llmops_router)


# Request models
class QAReq(BaseModel):
    context: str
    question: str

class TextReq(BaseModel):
    text: str

class ExplainReq(BaseModel):
    topic: str
    style: str = "detailed"  # options: detailed, step-by-step

class TranslateReq(BaseModel):
    text: str
    src_lang: str
    target_lang: str

@app.post("/question-answering")
async def question_answering(req: QAReq):
    """Question answering endpoint (long-context support)."""
    try:
        start = timeit()
        out = answer_question(req.context, req.question)
        latency = (timeit() - start) * 1000
        # record answer length (if available) and success
        try:
            record_metric("/question-answering", latency, {"success": True, "answer_len": len(out.get("answer", ""))})
        except Exception:
            pass
        return out
    except Exception as e:
        # record failure best-effort
        try:
            record_metric("/question-answering", 0.0, {"success": False, "error": str(e)})
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize-text")
async def summarize_text_endpoint(req: TextReq):
    """Text summarization endpoint."""
    try:
        start = timeit()
        s = summarize_text(req.text)
        latency = (timeit() - start) * 1000
        try:
            record_metric("/summarize-text", latency, {"success": True, "summary_len": len(s)})
        except Exception:
            pass
        return {"summary": s}
    except Exception as e:
        try:
            record_metric("/summarize-text", 0.0, {"success": False, "error": str(e)})
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/translate-text')
async def translate_text_endpoint(req: TranslateReq):
    """
    Translate text to a target language.
    Caller provides src_lang and target_lang codes; this endpoint maps
    short codes (e.g. 'hi', 'eng') to the internal NLLB-style codes.
    """
    start = timeit()
    try:
        text = req.text.strip()
        src_lang = req.src_lang.strip()
        target_lang = req.target_lang.strip()

        # Map short language codes to the NLLB / internal codes used by the
        # translation pipeline. If a code is not recognized, pass it through
        # unchanged so callers can use other codes.
        lang_map = {
            "hi": "hin_Deva",
            "eng": "eng_Latn",
        }

        mapped_src = lang_map.get(src_lang, src_lang)
        mapped_tgt = lang_map.get(target_lang, target_lang)

        translated_text = translate_text(text=text, src_lang=mapped_src, target_lang=mapped_tgt)

        latency = (timeit() - start) * 1000
        try:
            record_metric('/translate-text', latency, {'success': True, 'out_len': len(translated_text)})
        except Exception:
            pass
        return {'translation': translated_text}
    except Exception as e:
        try:
            record_metric('/translate-text', 0.0, {"success": False, "error": str(e)})
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/check-item-return-eligibility')
async def check_item_return_eligibility(file: UploadFile = File(...)):
    """Check whether an uploaded item image is eligible for return.

    Runs defect detection and returns a simple eligibility signal and
    confidence. This keeps the endpoint focused and clearly named for
    business logic that checks returns/refunds eligibility.
    """
    start = timeit()
    try:
        img_bytes = await file.read()

        # Step 1: Detect defects
        detection = detect_defect(img_bytes)
        print("🩻 Detection result:", detection)

        latency = (timeit() - start) * 1000
        try:
            record_metric('check-item-return-eligibility', latency, {"success": True, "confidence": detection.get("confidence")})
        except Exception:
            pass

        return {
            "detection": detection
        }
    except Exception as e:
        try:
            record_metric('check-item-return-eligibility', 0.0, {"success": False, "error": str(e)})
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/audio-transcribe")
async def audio_transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio (wav/mp3) to text."""
    start = timeit()
    try:
        audio_bytes = await file.read()
        transcription = transcribe_audio(audio_bytes)
        latency = (timeit() - start) * 1000
        try:
            record_metric("/audio-transcribe", latency, {"success": True, "out_len": len(transcription) if isinstance(transcription, str) else 0})
        except Exception:
            pass
        return transcription
    except Exception as e:
        try:
            record_metric("/audio-transcribe", 0.0, {"success": False, "error": str(e)})
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio-transcribe-translate")
async def audio_transcribe_translate(file: UploadFile = File(...)):
    """Transcribe and translate uploaded audio to the target language."""
    start = timeit()
    try:
        audio_bytes = await file.read()
        translated_text = transcribe_and_translate_audio(audio_bytes)
        latency = (timeit() - start) * 1000
        try:
            record_metric("/audio-transcribe-translate", latency, {"success": True, "out_len": len(translated_text) if isinstance(translated_text, str) else 0})
        except Exception:
            pass
        return {'translation': translated_text}
    except Exception as e:
        try:
            record_metric("/audio-transcribe-translate", 0.0, {"success": False, "error": str(e)})
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "CustomerAssist_finetune AI API is running successfully!"}
