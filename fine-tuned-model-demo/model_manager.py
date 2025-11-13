# File: fine-tuned-model-demo/model_manager.py

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

# --- Global cache for the loaded model ---
model = None
tokenizer = None
device = None
# ----------------------------------------

def load_model(base_model_id: str, adapter_dir: str):
    """
    Loads the fine-tuned model and tokenizer.
    This function populates the global 'model', 'tokenizer', and 'device' variables.
    """
    global model, tokenizer, device

    print(f"Starting model load. Base: {base_model_id}, Adapter: {adapter_dir}...")
    adapter_path = Path(adapter_dir)
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    # 1. Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Force float32 for stability, as we did in training
    model_dtype = torch.float32
    print(f"Using device: {device} with dtype: {model_dtype}")

    # 2. Load Tokenizer
    # We load the tokenizer from the adapter dir where it was saved
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {adapter_path}, falling back to base model ID... {e}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    print("Tokenizer loaded.")

    # 3. Load Base Model
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=False,  # Use built-in model code
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager" # Force stable attention
    )
    print("Base model loaded.")

    # 4. Resize base model embeddings (CRITICAL FIX)
    # This must be done *before* loading the adapter to match vocab size
    print(f"Resizing base model embeddings to {len(tokenizer)}...")
    base_model.resize_token_embeddings(len(tokenizer))

    # 5. Load and apply the PEFT LoRA Adapter
    try:
        print("Applying PEFT (LoRA) adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=model_dtype,
            device_map=None 
        )
        print("PEFT adapter applied successfully.")
    except Exception as e:
        print(f"Failed to load PEFT adapter: {e}. Using base model only.")
        model = base_model

    # 6. Move to device and set to evaluation mode
    model.to(device)
    model.eval()
    print(f"Model successfully loaded and moved to {device}.")