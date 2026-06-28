"""
Feedback-loop retraining script.
Reads data/feedback_log.jsonl, converts 'wrong' corrections into new QA training samples,
then runs a short QLoRA fine-tuning pass on the existing adapter to incorporate the corrections.

Usage:
    python retrain_from_feedback.py [--min-corrections 5] [--epochs 1]
"""

import os, json, sys, argparse, time
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE     = os.path.join(BASE_DIR, "data", "feedback_log.jsonl")
LLM_BASE_PATH     = os.path.join(BASE_DIR, "models", "tinyllama-1.1b-chat")
LLM_ADAPTER_PATH  = os.path.join(BASE_DIR, "models", "tinyllama-cassava-finetuned")
RETRAIN_OUT       = os.path.join(BASE_DIR, "models", "tinyllama-cassava-finetuned-v2")

SYSTEM_PROMPT = (
    "You are an expert agricultural assistant specializing in cassava diseases in Africa. "
    "Provide clear, concise advice to farmers about cassava disease symptoms, causes, and treatments."
)

DISEASE_QUESTIONS = {
    "mosaic_disease":       "What are the symptoms and treatment for Cassava Mosaic Disease?",
    "bacterial_blight":     "What are the symptoms and treatment for Cassava Bacterial Blight?",
    "brown_streak_disease": "What are the symptoms and treatment for Cassava Brown Streak Disease?",
    "green_mottle":         "What are the symptoms and treatment for Cassava Green Mottle?",
    "healthy":              "How do I keep my cassava plant healthy?",
}

DISEASE_ANSWERS = {
    "mosaic_disease":       "Cassava Mosaic Disease (CMD) causes yellowing and mottling of leaves, stunted growth, and reduced yield. It is spread by whiteflies feeding on infected plants. Use virus-free planting material, plant resistant varieties, and control whitefly populations through integrated pest management.",
    "bacterial_blight":     "Cassava Bacterial Blight (CBB) causes wilting, leaf blight, and stem rot. It spreads through infected cuttings and water splash. Use clean, disease-free planting material, avoid wounding plants, practice crop rotation, and remove infected plants immediately.",
    "brown_streak_disease": "Cassava Brown Streak Disease (CBSD) causes yellow leaf patches and brown necrotic streaks in the tuber. It is spread by whiteflies. Use certified virus-free cuttings, plant resistant varieties, and control whiteflies.",
    "green_mottle":         "Cassava Green Mottle causes mottling and distortion of leaves. Control involves using resistant varieties, certified planting material, and controlling insect vectors including whiteflies.",
    "healthy":              "Healthy cassava plants have bright green, uniform leaves with no spots or distortion. Maintain health through proper spacing (1 m x 1 m), regular weeding, balanced fertilization with NPK, and monitoring for early signs of disease or pest infestation.",
}


def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    records = []
    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_training_samples(records):
    """
    Convert 'wrong' feedback corrections into QA fine-tuning samples.
    Each correction teaches the model what the right disease actually is for
    questions it previously answered incorrectly.
    """
    samples = []
    for r in records:
        if r.get("vote") != "wrong":
            continue
        correct = r.get("correct_disease")
        if not correct or correct not in DISEASE_QUESTIONS:
            continue
        q = DISEASE_QUESTIONS[correct]
        a = DISEASE_ANSWERS[correct]
        text = (
            f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
            f"<|user|>\n{q}</s>\n"
            f"<|assistant|>\n{a}</s>"
        )
        samples.append({"text": text})
        # also add negative reinforcement — what was wrongly predicted
        wrong_pred = r.get("predicted_disease", "").replace("Cassava ", "").replace(" ", "_").lower()
        if wrong_pred in DISEASE_QUESTIONS:
            samples.append({
                "text": (
                    f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
                    f"<|user|>\n{DISEASE_QUESTIONS[wrong_pred]}</s>\n"
                    f"<|assistant|>\n{DISEASE_ANSWERS.get(wrong_pred, 'I need more information.')}</s>"
                )
            })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-corrections", type=int, default=5,
                        help="Minimum number of corrections before retraining (default: 5)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of fine-tuning epochs (default: 1)")
    args = parser.parse_args()

    print("Loading feedback log...")
    records = load_feedback()
    corrections = [r for r in records if r.get("vote") == "wrong"]
    print(f"  Total feedback: {len(records)} | Corrections: {len(corrections)}")

    if len(corrections) < args.min_corrections:
        print(f"  Not enough corrections ({len(corrections)} < {args.min_corrections}). Skipping retraining.")
        sys.exit(0)

    samples = build_training_samples(records)
    print(f"  Training samples generated: {len(samples)}")
    if not samples:
        print("  No valid samples. Skipping.")
        sys.exit(0)

    dataset = Dataset.from_list(samples)
    print(f"\nLoading base model + existing adapter...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(LLM_BASE_PATH, dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base, LLM_ADAPTER_PATH)
    model = model.merge_and_unload()  # merge old adapter before adding new one
    model = prepare_model_for_kbit_training(model) if torch.cuda.is_available() else model

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    sft_args = SFTConfig(
        output_dir=RETRAIN_OUT,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="epoch",
        fp16=False, bf16=False,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=dataset)
    print(f"\nRetraining on {len(samples)} feedback samples for {args.epochs} epoch(s)...")
    t0 = time.time()
    trainer.train()
    trainer.save_model(RETRAIN_OUT)
    tokenizer.save_pretrained(RETRAIN_OUT)
    print(f"\nDone! New adapter saved to {RETRAIN_OUT} ({round((time.time()-t0)/60, 1)} min)")
    print("To use the new model, update LLM_ADAPTER_PATH in pipeline.py to point to", RETRAIN_OUT)


if __name__ == "__main__":
    main()
