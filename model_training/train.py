import os
import torch
import re
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model
from pathlib import Path

# --- 1. Settings ---
MODEL_ID = "distilgpt2"
DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
# OUTPUT_DIR = "../models/distilgpt2_finetuned_adapter"
BASE_DIR = Path(__file__).resolve().parent.parent  # goes up one level
OUTPUT_DIR = BASE_DIR / "models" / "distilgpt2_finetuned_adapter"

EPOCHS = 3
BATCH_SIZE = 1 
MAX_LENGTH = 256
USE_LORA = True

print(f"--- Settings ---")
print(f"MODEL_ID: {MODEL_ID}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"---")

# --- 2. Device Setup ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if torch.cuda.is_available():
    device = 'cuda'
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# --- FIX 1: Force float32 for training stability ---
model_dtype = torch.float32 
use_pin_memory = True if device == 'cuda' else False

print(f'Using device: {device} (model dtype={model_dtype})')

# --- 3. Dataset Helpers (from your notebook) ---
def safe_load_customer_dataset(dataset_id):
    try:
        ds = load_dataset(dataset_id)
        return ds.get('train', ds)
    except Exception as e:
        print(f'Could not load dataset {dataset_id}: {e}')
        print('Falling back to a tiny synthetic customer support dataset.')
        samples = [
            {'customer': "My order hasn't arrived, it's been 10 days.", 'agent': "I'm sorry. Can you share your order id?"},
            {'customer': 'I was charged twice for the same order.', 'agent': "I can help. Please share the transaction id."},
        ]
        return Dataset.from_list(samples)

def build_prompt_for_training(row):
    # Using the Human/Assistant format
    if 'customer' in row and 'agent' in row:
        return {'text': f"Human: {row['customer']}\nAssistant: {row['agent']}\n"}
    if 'input' in row and 'output' in row:
        return {'text': f"Human: {row['input']}\nAssistant: {row['output']}\n"}
    if 'instruction' in row and 'response' in row:
         return {'text': f"Human: {row['instruction']}\nAssistant: {row['response']}\n"}
    return {'text': str(row)}

# --- 4. Load and Prepare Dataset ---
print("Loading dataset...")
raw_ds = safe_load_customer_dataset(DATASET_ID)
ds = raw_ds.map(build_prompt_for_training)

# Split and reduce for a local run
if len(ds) > 2000:
    ds_split = ds.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_ds = ds_split['train'].select(range(min(4096, len(ds_split['train']))))
    eval_ds = ds_split['test'].select(range(min(128, len(ds_split['test']))))
else:
    ds_split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = ds_split['train']
    eval_ds = ds_split['test']

print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

# --- 5. Load Model and Tokenizer (Only ONCE) ---
print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) 
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
    torch_dtype=model_dtype,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)
model.resize_token_embeddings(len(tokenizer))
print("Base model loaded.")

# --- 6. Apply LoRA/PEFT ---
if USE_LORA:
    print("Applying LoRA adapter (PEFT)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"], # Correct for distilgpt2
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adapter applied.")
else:
    print("Skipping LoRA. Training full model (slower).")

# --- 7. Tokenize Dataset ---
def tokenize_for_lm(examples):
    texts = [str(t) for t in examples.get('text', [])]
    outputs = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

print("Tokenizing datasets...")
train_tok = train_ds.map(tokenize_for_lm, batched=True, remove_columns=train_ds.column_names)
eval_tok = eval_ds.map(tokenize_for_lm, batched=True, remove_columns=eval_ds.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- 8. Define Trainer ---
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=10,
    save_total_limit=2,
    # --- FIX 2: Disable fp16 ---
    fp16=False, 
    remove_unused_columns=False,
    dataloader_pin_memory=use_pin_memory,
    use_mps_device=(device == 'mps'), 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
)

# --- 9. !! RUN TRAINING AND SAVE MODEL !! ---
print("--- Starting Training ---")
trainer.train()
print("--- Training Complete ---")

# Save the final adapter
print(f"Saving fine-tuned adapter to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Adapter saved successfully.")


# --- 10. Test the Fine-Tuned Model ---
print("\n--- Testing the Fine-Tuned Model ---")

del model
if device == 'cuda':
    torch.cuda.empty_cache()
elif device == 'mps':
    torch.mps.empty_cache()


from peft import PeftModel

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
    torch_dtype=model_dtype,
    attn_implementation="eager"
)
# 2. Load the tokenizer from our saved directory
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

# Resize the base model embeddings *before* loading the adapter
base_model.resize_token_embeddings(len(tokenizer))

# 3. Apply the adapter
model = PeftModel.from_pretrained(
    base_model,
    OUTPUT_DIR,
    torch_dtype=model_dtype,
)
model = model.to(device)
model.eval()

print("Fine-tuned model reloaded for testing.")

# Test generation
prompt = "Human: I haven't received my refund after 10 days. What should I do?\nAssistant:"
enc = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)

with torch.no_grad():
    gen_kwargs = dict(
        max_new_tokens=80,
        do_sample=True,
        temperature=0.22,
        top_k=40,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        **gen_kwargs
    )

gen_tokens = out[0][enc["input_ids"].shape[-1]:]
reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

print(f"\nPROMPT:\n{prompt}")
print(f"\nMODEL REPLY:\n{reply}")