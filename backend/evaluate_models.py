"""
Evaluate both trained models and compare against baselines.
- MobileNetV2: accuracy, per-class F1, confusion matrix
- TinyLlama fine-tuned vs baseline: manual Q&A scoring (same 10 test questions)
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
IMAGE_MODEL_PATH = "models/cassava_classifier.pth"
LLM_BASE_PATH = "models/tinyllama-1.1b-chat"
LLM_FINETUNED_PATH = "models/tinyllama-cassava-finetuned"
DATA_DIR = "data/cassava-dataset/data"
BASELINE_PATH = "data/baseline_evaluation.json"
RESULTS_PATH = "data/evaluation_results.json"

IMG_SIZE = 224
BATCH_SIZE = 32

DISEASE_MAP = {
    "Cassava___bacterial_blight": "CBB",
    "Cassava___brown_streak_disease": "CBSD",
    "Cassava___green_mottle": "CGM",
    "Cassava___healthy": "Healthy",
    "Cassava___mosaic_disease": "CMD",
}

TEST_QUESTIONS = [
    {"q": "What are the symptoms of cassava mosaic disease?",
     "expected_keywords": ["yellow", "mosaic", "distort", "leaf", "stunt"]},
    {"q": "How do I treat bacterial blight in cassava?",
     "expected_keywords": ["clean", "resistant", "sanit", "remove", "variety"]},
    {"q": "What causes cassava brown streak disease?",
     "expected_keywords": ["virus", "whitefly", "CBSD", "streak"]},
    {"q": "How can I prevent cassava green mottle?",
     "expected_keywords": ["resist", "plant", "free", "certif", "whitefly"]},
    {"q": "When should I harvest cassava?",
     "expected_keywords": ["month", "mature", "tuber", "stem", "leaf"]},
    {"q": "What is the best spacing for planting cassava?",
     "expected_keywords": ["meter", "spacing", "1m", "3ft", "row"]},
    {"q": "How do whiteflies spread disease in cassava?",
     "expected_keywords": ["virus", "transmit", "spread", "vector", "feed"]},
    {"q": "What fertilizer is good for cassava?",
     "expected_keywords": ["nitrogen", "phosphorus", "potassium", "NPK", "manure", "organic"]},
    {"q": "How do I identify healthy cassava leaves?",
     "expected_keywords": ["green", "no spot", "healthy", "uniform", "smooth"]},
    {"q": "What are disease-resistant cassava varieties?",
     "expected_keywords": ["TME", "variety", "resist", "IITA", "breed"]},
]


# ── Image model evaluation ───────────────────────────────────────────────────
def evaluate_image_model():
    print("\n=== MobileNetV2 Image Classifier ===")
    if not os.path.exists(IMAGE_MODEL_PATH):
        print(f"Model not found at {IMAGE_MODEL_PATH}. Train it first with train_image_classifier.py")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    arch = checkpoint.get("model_arch", "mobilenet_v2")
    img_size = checkpoint.get("img_size", IMG_SIZE)

    if arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4, inplace=True),
            nn.Linear(in_features, len(class_names)),
        )
    else:
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(class_names)),
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)
    # Use last 20% as val (same seed as training)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    short_names = [DISEASE_MAP.get(c, c) for c in class_names]
    report = classification_report(all_labels, all_preds, target_names=short_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=short_names))

    return {
        "accuracy": round(float(acc), 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": short_names,
        "best_epoch_val_acc": checkpoint.get("val_acc", None),
    }


# ── LLM evaluation ───────────────────────────────────────────────────────────
def score_response(response: str, expected_keywords: list) -> float:
    resp_lower = response.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in resp_lower)
    return round(hits / len(expected_keywords), 2)


def generate_answer(model, tokenizer, question: str, max_new_tokens=150) -> str:
    SYSTEM_PROMPT = (
        "You are an expert agricultural assistant specializing in cassava diseases in Africa. "
        "Provide clear, concise advice to farmers about cassava disease symptoms, causes, and treatments."
    )
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def evaluate_llm():
    print("\n=== TinyLlama LLM Evaluation ===")
    if not os.path.exists(LLM_FINETUNED_PATH):
        print(f"Fine-tuned model not found at {LLM_FINETUNED_PATH}. Train it first with train_llm_qlora.py")
        return None

    tokenizer = AutoTokenizer.from_pretrained(LLM_FINETUNED_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Load fine-tuned model
    print("Loading fine-tuned TinyLlama...")
    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_BASE_PATH, torch_dtype=torch.float32, device_map="cpu"
    )
    finetuned_model = PeftModel.from_pretrained(base_model, LLM_FINETUNED_PATH)
    finetuned_model.eval()

    # Load baseline scores
    baseline_scores = []
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline_data = json.load(f)
        baseline_scores = [r.get("score", 0) for r in baseline_data.get("results", [])]

    finetuned_scores = []
    results = []

    print(f"\nEvaluating {len(TEST_QUESTIONS)} test questions...")
    for i, item in enumerate(TEST_QUESTIONS):
        answer = generate_answer(finetuned_model, tokenizer, item["q"])
        score = score_response(answer, item["expected_keywords"])
        finetuned_scores.append(score)
        baseline = baseline_scores[i] if i < len(baseline_scores) else None

        print(f"\nQ{i+1}: {item['q']}")
        print(f"  Answer: {answer[:120]}...")
        print(f"  Score: {score:.2f} (baseline: {baseline})")

        results.append({
            "question": item["q"],
            "answer": answer,
            "score": score,
            "baseline_score": baseline,
        })

    avg_finetuned = round(np.mean(finetuned_scores), 3)
    avg_baseline = round(np.mean(baseline_scores), 3) if baseline_scores else None

    print(f"\nAvg fine-tuned score: {avg_finetuned}")
    print(f"Avg baseline score:   {avg_baseline}")
    print(f"Improvement: {round(avg_finetuned - (avg_baseline or 0), 3)}")

    return {
        "avg_finetuned_score": avg_finetuned,
        "avg_baseline_score": avg_baseline,
        "improvement": round(avg_finetuned - (avg_baseline or 0), 3),
        "results": results,
    }


def main():
    print("Running full model evaluation...")

    image_results = evaluate_image_model()
    llm_results = evaluate_llm()

    output = {
        "image_classifier": image_results,
        "llm": llm_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nAll results saved to {RESULTS_PATH}")

    if image_results:
        print(f"Image classifier accuracy: {image_results['accuracy']:.4f}")
    if llm_results:
        print(f"LLM score: {llm_results['avg_finetuned_score']} (was {llm_results['avg_baseline_score']})")


if __name__ == "__main__":
    main()
