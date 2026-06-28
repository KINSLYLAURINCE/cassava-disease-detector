"""
TinyLlama 1.1B QLoRA fine-tuning on cassava Q&A dataset.
GPU-optimized for Google Colab (T4/A100). Falls back to CPU full-precision.
4-bit NF4 quantization on GPU cuts VRAM to ~3 GB (fits T4 16 GB easily).
Dataset: data/cassava_qa_dataset.json (3,542 instruction-response pairs)
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = DEVICE == "cuda"

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_PATH = "models/tinyllama-1.1b-chat"
DATASET_PATH = "data/cassava_qa_dataset.json"
OUTPUT_DIR = "models/tinyllama-cassava-finetuned"
RESULTS_PATH = "data/llm_training_results.json"

MAX_SEQ_LENGTH = 512        # doubled from 256 — captures longer responses
EPOCHS = 5                  # more epochs on GPU
LR = 2e-4
WARMUP_RATIO = 0.05
GRAD_ACCUM_STEPS = 4 if DEVICE == "cuda" else 8

# Batch: 4 on GPU (T4 16 GB), 1 on CPU
BATCH_SIZE = 4 if DEVICE == "cuda" else 1

# LoRA: expand to more modules and higher rank for better capacity
LORA_R = 32             # up from 16
LORA_ALPHA = 64         # 2× rank
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",   # MLP layers too
]

SYSTEM_PROMPT = (
    "You are an expert agricultural assistant specializing in cassava diseases in Africa. "
    "Provide clear, concise advice to farmers about cassava disease symptoms, causes, and treatments."
)


def format_sample(sample):
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{sample['instruction']}</s>\n"
        f"<|assistant|>\n{sample['response']}</s>"
    )


def load_dataset():
    with open(DATASET_PATH) as f:
        raw = json.load(f)
    raw = [s for s in raw if len(s.get("response", "")) > 10 and len(s.get("instruction", "")) > 5]
    print(f"Dataset: {len(raw)} samples after filtering")

    texts = [format_sample(s) for s in raw]
    split = int(0.9 * len(texts))
    train_ds = Dataset.from_dict({"text": texts[:split]})
    val_ds = Dataset.from_dict({"text": texts[split:]})
    return train_ds, val_ds


def main():
    print(f"Device: {DEVICE} | 4-bit QLoRA: {USE_4BIT}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA} | Seq len: {MAX_SEQ_LENGTH} | Epochs: {EPOCHS}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model (4-bit on GPU, fp32 on CPU) ─────────────────────────────────────
    print("Loading TinyLlama 1.1B...")
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nPreparing dataset...")
    train_ds, val_ds = load_dataset()
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(DEVICE == "cuda"),
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print(f"\nStarting QLoRA fine-tuning ({EPOCHS} epochs)...")
    train_result = trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to: {OUTPUT_DIR}")

    metrics = {
        "model_arch": "tinyllama-1.1b-qlora",
        "device": DEVICE,
        "use_4bit": USE_4BIT,
        "train_runtime_seconds": round(train_result.metrics.get("train_runtime", 0), 1),
        "train_loss": round(train_result.metrics.get("train_loss", 0), 4),
        "train_samples_per_second": round(train_result.metrics.get("train_samples_per_second", 0), 3),
        "epochs": EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_target_modules": LORA_TARGET_MODULES,
        "max_seq_length": MAX_SEQ_LENGTH,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "output_dir": OUTPUT_DIR,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n== Training complete ==")
    print(f"Train loss: {metrics['train_loss']}")
    print(f"Runtime   : {metrics['train_runtime_seconds']/3600:.1f} hours")
    print(f"Results   : {RESULTS_PATH}")


if __name__ == "__main__":
    main()
