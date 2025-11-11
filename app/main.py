"""REST API surface for Customer Assist.

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
from app.monitor import log_llm_interaction

app = FastAPI(title="Customer Assist AI", description="AI-powered Customer Service Assistant", version="1.0")

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
    endpoint = "/question-answering"
    try:
        start = timeit()
        out = answer_question(req.context, req.question)
        latency = (timeit() - start) * 1000
        
        # LLMOPS: Log interaction
        full_prompt = f"Context: {req.context.strip()} | Question: {req.question.strip()}"
        answer = out.get("answer", "")
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=full_prompt,
            output=answer,
            metadata={"answer_len": len(answer)}
        )
        
        record_metric(endpoint, latency, {"answer_len": len(answer)})
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize-text")
async def summarize_text_endpoint(req: TextReq):
    """Text summarization endpoint."""
    endpoint = "/summarize-text"
    try:
        start = timeit()
        # Assumes summarize_text returns (summary, prompt) from models.py
        s, prompt = summarize_text(req.text) 
        latency = (timeit() - start) * 1000
        
        # LLMOPS: Log interaction
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=prompt,
            output=s,
            metadata={"summary_len": len(s)}
        )
        
        record_metric(endpoint, latency, {"summary_len": len(s)})
        return {"summary": s}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/translate-text')
async def translate_text_endpoint(req: TranslateReq):
    """
    Translate text to a target language.
    Caller provides src_lang and target_lang codes; this endpoint maps
    short codes (e.g. 'hi', 'eng') to the internal NLLB-style codes.
    """
    endpoint = "/translate-text"
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
        
        # Assumes translate_text returns (translated_text, prompt) from models.py
        translated_text, prompt = translate_text(text=text, src_lang=mapped_src, target_lang=mapped_tgt)

        latency = (timeit() - start) * 1000
        
        # LLMOPS: Log interaction for translation
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=prompt, # The text being translated
            output=translated_text,
            metadata={"src": mapped_src, "tgt": mapped_tgt, 'out_len': len(translated_text)}
        )
        
        record_metric(endpoint, latency, {'out_len': len(translated_text)})
        return {'translation': translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/check-item-return-eligibility')
async def check_item_return_eligibility(file: UploadFile = File(...)):
    """Check whether an uploaded item image is eligible for return.
    Runs defect detection and returns a simple eligibility signal and
    confidence.
    """
    endpoint = "/check-item-return-eligibility"
    start = timeit()
    try:
        # Read a small portion of the file for logging context, not the whole image
        # This is a critical optimization for large file uploads in logging!
        file_name = file.filename
        content_type = file.content_type
        
        img_bytes = await file.read()

        # Step 1: Detect defects
        detection = detect_defect(img_bytes)
        print("🩻 Detection result:", detection)
        
        latency = (timeit() - start) * 1000

        # # Simple eligibility rule: if defective -> not eligible, else eligible.
        # eligibility = "not_eligible" if detection.get("is_defective") else "eligible"
        
        # LLMOPS: Log interaction for image classification/detection
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=f"Image file uploaded: {file_name} ({content_type})",
            output=str(detection), # Log the full detection dictionary as output
            metadata={"eligible": detection.get("eligible_for_return"), "predicted_label": detection.get("predicted_label"), "file_size_bytes": len(img_bytes)}
        )
        
        record_metric(endpoint, latency, {"eligible": detection.get("eligible_for_return"), "predicted_label": detection.get("predicted_label")})

        return detection
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/audio-transcribe")
async def audio_transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio (wav/mp3) to text."""
    endpoint = "/audio-transcribe"
    start = timeit()
    try:
        file_name = file.filename
        audio_bytes = await file.read()
        
        transcribed_result = transcribe_audio(audio_bytes)
        transcription = transcribed_result.get("transcription", "")
        
        latency = (timeit() - start) * 1000

        # LLMOPS: Log interaction for ASR
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=f"Audio file uploaded: {file_name}",
            output=transcription,
            metadata={"transcription_len": len(transcription), "file_size_bytes": len(audio_bytes)}
        )
        
        record_metric(endpoint, latency, {"transcription_len": len(transcription)})
        return transcribed_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio-transcribe-translate")
async def audio_transcribe_translate(file: UploadFile = File(...)):
    """Transcribe and translate uploaded audio to the target language."""
    endpoint = "/audio-transcribe-translate"
    start = timeit()
    try:
        file_name = file.filename
        audio_bytes = await file.read()
        
        # This function handles two model calls (ASR and Translation) internally
        translated_text = transcribe_and_translate_audio(audio_bytes)
        
        latency = (timeit() - start) * 1000

        # LLMOPS: Log interaction for ASR+Translation (using a compound prompt)
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=f"Audio file uploaded for ASR and Translation: {file_name}",
            output=translated_text,
            metadata={"translation_len": len(translated_text), "file_size_bytes": len(audio_bytes)}
        )

        record_metric(endpoint, latency, {"translation_len": len(translated_text)})
        return {'translation': translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "CustomerAssist_finetune AI API is running successfully!"}