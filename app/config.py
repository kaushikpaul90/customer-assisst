"""Configuration constants used across the demo app.

This module exposes a small list of model identifiers and path
constants. These are intentionally simple constants (strings and
Paths) so callers can import them without triggering heavy work.
"""

import torch

# Models (Hugging Face names) -------------------------------------------------
QA_MODEL = "deepset/roberta-base-squad2"
SUMMARIZER = "philschmid/bart-large-cnn-samsum"
EXPLAINER = "google/flan-t5-large"
TRANSLATOR = "facebook/nllb-200-distilled-600M"

# Image models
IMG_CLASSIFIER = "google/vit-base-patch16-224"
IMAGE_CAPTION = "Salesforce/blip-image-captioning-base"
DEFECT_DETECTOR = "openai/clip-vit-large-patch14"

# ASR (Whisper style model id)
ASR_MODEL = "openai/whisper-medium"

# OCR languages (easyocr)
OCR_LANGS = ["en"]

# Compute device (string) - macOS uses MPS (Metal Performance Shaders) for GPU,
# falls back to CPU otherwise. CUDA is not available on macOS.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ----------------------------
# Pipeline Task Names
# ----------------------------
TASK_QA = "question-answering"
TASK_SUMMARIZATION = "summarization"
TASK_TEXT_GENERATION = "text-generation"
TASK_IMAGE_CLASSIFICATION = "image-classification"
TASK_IMAGE_CAPTION = "image-to-text"
TASK_ASR = "automatic-speech-recognition"

# ----------------------------
# Pipeline Device Configuration
# ----------------------------
# Note: transformers pipeline() uses -1 for CPU and 0 for GPU
# On macOS with MPS, we use CPU device (-1) as transformers doesn't support MPS directly
DEVICE_ID = -1  # CPU device for transformers pipelines
DEVICE_MAP = "cpu"  # Use CPU device mapping for transformers models

# ----------------------------
# Pipeline Keyword Arguments
# ----------------------------
QA_PIPELINE_KWARGS = {
    "device": DEVICE_ID
}

SUMMARIZER_PIPELINE_KWARGS = {
    "device": DEVICE_ID
}

EXPLAINER_PIPELINE_KWARGS = {
    "model_kwargs": {"torch_dtype": torch.bfloat16},
    "device_map": DEVICE_MAP
}

IMG_CLASSIFIER_PIPELINE_KWARGS = {
    "device": DEVICE_ID
}

IMAGE_CAPTIONER_PIPELINE_KWARGS = {
    "device": DEVICE_ID
}

ASR_PIPELINE_KWARGS = {
    "return_timestamps": True,
    "device": DEVICE_ID  # Use CPU device on macOS
}
