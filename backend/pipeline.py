"""
Cassava Disease Detector — Inference Pipeline
1. EfficientNet-B4 classifies the leaf image → disease label + confidence
2. TinyLlama QLoRA gives the farmer actionable advice about the detected disease
3. DuckDuckGo search enriches the response with up-to-date web results
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from ddgs import DDGS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "cassava_efficientnet_b4.pth")
LLM_BASE_PATH    = os.path.join(BASE_DIR, "models", "tinyllama-1.1b-chat")
LLM_ADAPTER_PATH = os.path.join(BASE_DIR, "models", "tinyllama-cassava-finetuned")

DISEASE_LABELS = {
    "bacterial_blight":   "Cassava Bacterial Blight (CBB)",
    "brown_streak_disease": "Cassava Brown Streak Disease (CBSD)",
    "green_mottle":       "Cassava Green Mottle (CGM)",
    "healthy":            "Healthy",
    "mosaic_disease":     "Cassava Mosaic Disease (CMD)",
}


SYSTEM_PROMPT = (
    "You are an expert agricultural assistant specializing in cassava diseases in Africa. "
    "Provide clear, concise advice to farmers about cassava disease symptoms, causes, and treatments."
)

# Per-disease targeted questions matching the training data style
DISEASE_QUESTIONS = {
    "mosaic_disease": [
        ("Symptoms",   "what are the signs and symptoms of cassava mosaic disease?"),
        ("Treatment",  "how do you control or treat cassava mosaic disease?"),
        ("Prevention", "how can a farmer prevent cassava mosaic disease?"),
    ],
    "bacterial_blight": [
        ("Symptoms",   "what are the signs and symptoms of cassava bacterial blight?"),
        ("Treatment",  "how do you control or treat cassava bacterial blight?"),
        ("Prevention", "how can a farmer prevent cassava bacterial blight?"),
    ],
    "brown_streak_disease": [
        ("Symptoms",   "what are the signs and symptoms of cassava brown streak disease?"),
        ("Treatment",  "how do you control cassava brown streak disease?"),
        ("Prevention", "how can a farmer prevent cassava brown streak disease?"),
    ],
    "green_mottle": [
        ("Symptoms",   "what are the signs and symptoms of cassava green mottle?"),
        ("Treatment",  "how do you treat or control cassava green mottle?"),
        ("Prevention", "how can a farmer prevent cassava green mottle?"),
    ],
    "healthy": [
        ("Status",     "how do you identify a healthy cassava plant?"),
        ("Harvest",    "when should a farmer harvest cassava?"),
    ],
}


# Accurate disease-specific treatment and prevention info
# (The LLM is weak on differentiating treatments — use curated knowledge here)
TREATMENT_INFO = {
    "mosaic_disease": {
        "treatment": "There is no cure for CMD once a plant is infected. Remove and burn all infected plants immediately to prevent spread. Replant with virus-free cuttings from certified sources. Apply systemic insecticides (e.g., imidacloprid) to control whitefly vectors.",
        "prevention": "Plant CMD-resistant varieties (e.g., TME 204, NASE 14, MM96). Use clean planting material from certified nurseries. Control whiteflies with yellow sticky traps, neem-based sprays, or insecticides. Maintain a 10-meter isolation distance from infected fields. Practice early planting to avoid peak whitefly season.",
    },
    "bacterial_blight": {
        "treatment": "No chemical treatment is effective against CBB. Remove and destroy all infected plants by burning. Disinfect cutting tools with 10% bleach solution between plants. Apply copper-based bactericides (e.g., Bordeaux mixture) as a preventive spray on healthy plants nearby.",
        "prevention": "Use disease-free stem cuttings from healthy plants. Practice crop rotation (2-3 years without cassava). Avoid working in fields when plants are wet. Plant resistant varieties (e.g., TMS 30572, TME 7). Maintain good field drainage to reduce humidity.",
    },
    "brown_streak_disease": {
        "treatment": "No cure exists for CBSD. Uproot and burn all symptomatic plants immediately. Do not use tubers from infected plants as planting material. Replace with certified virus-free cuttings. Monitor fields every 2 weeks for new symptoms.",
        "prevention": "Plant CBSD-tolerant varieties (e.g., NASE 19, Kiroba, NDL 2006). Source planting material only from certified disease-free nurseries. Control whitefly populations using integrated pest management (IPM). Inspect new plantings at 3, 6, and 9 months. Avoid planting near older infected fields.",
    },
    "green_mottle": {
        "treatment": "Remove and destroy infected plants immediately. There is no chemical cure. Ensure neighboring healthy plants receive proper nutrition (NPK fertilizer) to boost resistance. Monitor surrounding plants weekly for new symptoms.",
        "prevention": "Use certified virus-free planting material. Control whitefly and aphid vectors using neem oil sprays or insecticidal soap. Plant resistant varieties where available. Maintain field hygiene by removing weeds that can harbor the virus. Avoid planting cassava near other infected solanaceous crops.",
    },
    "healthy": {
        "treatment": "No treatment needed. Continue good agricultural practices to maintain plant health.",
        "prevention": "Maintain proper plant spacing (1m x 1m). Apply balanced NPK fertilizer (15-15-15) at planting and 8 weeks after. Weed regularly, especially in the first 3 months. Monitor weekly for early signs of disease. Harvest between 8-12 months after planting depending on variety.",
    },
}

ADVICE_CACHE_PATH = os.path.join(BASE_DIR, "data", "advice_cache.json")


class CassavaPipeline:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._image_model = None
        self._class_names = None
        self._llm = None
        self._tokenizer = None
        self._img_size = 380  # EfficientNet-B4 default
        self._advice_cache = self._load_advice_cache()

    def _load_advice_cache(self):
        if os.path.exists(ADVICE_CACHE_PATH):
            with open(ADVICE_CACHE_PATH) as f:
                return json.load(f)
        return {}

    def _save_advice_cache(self):
        os.makedirs(os.path.dirname(ADVICE_CACHE_PATH), exist_ok=True)
        with open(ADVICE_CACHE_PATH, "w") as f:
            json.dump(self._advice_cache, f, indent=2)

    def warm_advice_cache(self, force: bool = False):
        """Pre-generate advice for all 5 diseases and cache to disk. Call once at startup."""
        self._load_llm()
        updated = False
        for disease in DISEASE_QUESTIONS:
            if force or disease not in self._advice_cache:
                print(f"  Caching advice for: {disease}")
                self._advice_cache[disease] = self._generate_rich_advice(disease)
                updated = True
        if updated:
            self._save_advice_cache()
            print("  Advice cache saved.")

    def _generate_rich_advice(self, disease_label: str) -> str:
        """Generate advice using LLM for symptoms + hardcoded accurate treatment/prevention."""
        symptoms = self._ask(DISEASE_QUESTIONS[disease_label][0][1])
        treatment = TREATMENT_INFO.get(disease_label, {}).get("treatment", "")
        prevention = TREATMENT_INFO.get(disease_label, {}).get("prevention", "")
        parts = [f"Symptoms: {symptoms}"]
        if treatment:
            parts.append(f"Treatment: {treatment}")
        if prevention:
            parts.append(f"Prevention: {prevention}")
        return "\n\n".join(parts)

    # ── lazy loaders ──────────────────────────────────────────────────────────

    def _load_image_model(self):
        if self._image_model is not None:
            return
        print("Loading image classifier...")
        checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=self.device, weights_only=False)
        # support both key naming conventions
        label_map = checkpoint.get("label_map") or checkpoint.get("class_names")
        if isinstance(label_map, dict):
            # {idx: name} → sorted list
            self._class_names = [label_map[i] for i in sorted(label_map)]
        else:
            self._class_names = label_map
        self._img_size = checkpoint.get("img_size", 380)

        arch = checkpoint.get("arch") or checkpoint.get("model_arch", "efficientnet_b4")
        if arch == "efficientnet_b4":
            model = models.efficientnet_b4(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.4, inplace=True),
                nn.Linear(in_features, len(self._class_names)),
            )
        else:
            model = models.mobilenet_v2(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.last_channel, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, len(self._class_names)),
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self._image_model = model
        print(f"  Image model ready ({arch}, {len(self._class_names)} classes)")

    def _load_llm(self):
        if self._llm is not None:
            return
        print("Loading TinyLlama + LoRA adapter...")
        self._tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_PATH)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            LLM_BASE_PATH, dtype=torch.float32, device_map="cpu"
        )
        self._llm = PeftModel.from_pretrained(base, LLM_ADAPTER_PATH)
        self._llm.eval()
        print("  LLM ready")

    # ── leaf pre-check ─────────────────────────────────────────────────────────

    def is_leaf_image(self, image_path: str, green_threshold: float = 0.15) -> bool:
        """
        Quick check: is this image likely a plant leaf?
        Uses HSV color space to measure green pixel ratio.
        Cassava leaves (healthy or diseased) always have significant green/yellow-green content.
        A dog, car, or random object typically has < 15% green pixels.
        """
        img = Image.open(image_path).convert("RGB")
        img_small = img.resize((128, 128))
        arr = np.array(img_small, dtype=np.float32) / 255.0

        # Convert RGB to HSV manually
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        diff = cmax - cmin

        # Hue calculation
        hue = np.zeros_like(cmax)
        mask = diff > 0.01
        # green dominant
        green_mask = mask & (cmax == g)
        hue[green_mask] = 60 * (((b[green_mask] - r[green_mask]) / diff[green_mask]) + 2)
        # red dominant
        red_mask = mask & (cmax == r)
        hue[red_mask] = 60 * (((g[red_mask] - b[red_mask]) / diff[red_mask]) % 6)
        # blue dominant
        blue_mask = mask & (cmax == b)
        hue[blue_mask] = 60 * (((r[blue_mask] - g[blue_mask]) / diff[blue_mask]) + 4)

        # Saturation
        sat = np.zeros_like(cmax)
        sat[cmax > 0] = diff[cmax > 0] / cmax[cmax > 0]

        # Green/yellow-green range: hue 35-155, saturation > 0.15, brightness > 0.12
        # Excludes browns (hue < 35) and blues (hue > 155)
        leaf_pixels = ((hue >= 35) & (hue <= 155) & (sat > 0.15) & (cmax > 0.12))
        green_ratio = float(leaf_pixels.sum()) / (128 * 128)

        return green_ratio >= green_threshold

    # ── image classification ──────────────────────────────────────────────────

    def classify(self, image_path: str) -> dict:
        self._load_image_model()

        transform = transforms.Compose([
            transforms.Resize((self._img_size, self._img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._image_model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            top_idx = probs.argmax().item()

        top_conf = float(probs[top_idx])

        raw_label = self._class_names[top_idx]
        short_key = raw_label.replace("Cassava___", "").lower()
        display   = DISEASE_LABELS.get(short_key, raw_label)

        top5 = sorted(
            [(self._class_names[i].replace("Cassava___",""), float(probs[i])) for i in range(len(probs))],
            key=lambda x: x[1], reverse=True
        )

        return {
            "label":      short_key,
            "display":    display,
            "confidence": round(top_conf, 4),
            "is_cassava": True,
            "top5":       [(name, round(p, 4)) for name, p in top5],
        }

    # ── LLM advice ────────────────────────────────────────────────────────────

    def _ask(self, question: str, max_new_tokens: int = 60) -> str:
        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
            f"<|user|>\n{question}</s>\n"
            f"<|assistant|>\n"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def advise(self, disease_label: str, question: str = None, max_new_tokens: int = 60, _bypass_cache: bool = False) -> str:
        self._load_llm()

        # Custom single question — always answer live
        if question is not None:
            return self._ask(question, max_new_tokens)

        # Return cached advice instantly if available
        if not _bypass_cache and disease_label in self._advice_cache:
            return self._advice_cache[disease_label]

        # Generate fresh advice
        questions = DISEASE_QUESTIONS.get(disease_label)
        if not questions:
            return self._ask(
                f"what should a farmer do if their cassava has {DISEASE_LABELS.get(disease_label, disease_label)}?",
                max_new_tokens
            )

        sections = []
        for label, q in questions:
            answer = self._ask(q, max_new_tokens)
            sections.append(f"{label}: {answer}")

        return "\n\n".join(sections)

    # ── web search ────────────────────────────────────────────────────────────

    def web_search(self, disease_label: str, max_results: int = 3) -> list:
        disease_name = DISEASE_LABELS.get(disease_label, disease_label)
        query = f"{disease_name} cassava treatment prevention farmer guide"
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [
                {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
                for r in results
            ]
        except Exception as e:
            print(f"  Web search failed: {e}")
            return []

    # ── full pipeline ─────────────────────────────────────────────────────────

    def run(self, image_path: str, question: str = None, search: bool = True) -> dict:
        print(f"\nAnalyzing: {image_path}")

        if not self.is_leaf_image(image_path):
            print("  Rejected: image does not appear to be a plant leaf.")
            return {
                "image":       image_path,
                "disease":     None,
                "confidence":  0.0,
                "is_cassava":  False,
                "top5":        [],
                "advice":      None,
                "web_results": [],
                "error":       "This image does not appear to be a cassava leaf. Please upload a clear photo of a cassava leaf.",
            }

        classification = self.classify(image_path)
        print(f"  Detected: {classification['display']} ({classification['confidence']*100:.1f}%)")

        print("  Generating advice...")
        advice = self.advise(classification["label"], question)

        web_results = []
        if search:
            print("  Searching web for more info...")
            web_results = self.web_search(classification["label"])

        return {
            "image":       image_path,
            "disease":     classification["display"],
            "confidence":  classification["confidence"],
            "is_cassava":  True,
            "top5":        classification["top5"],
            "advice":      advice,
            "web_results": web_results,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse, json

    parser = argparse.ArgumentParser(description="Cassava Disease Detector")
    parser.add_argument("image", help="Path to cassava leaf image")
    parser.add_argument("--question", "-q", default=None, help="Custom follow-up question for the LLM")
    parser.add_argument("--classify-only", action="store_true", help="Skip LLM, only classify the image")
    parser.add_argument("--no-search", action="store_true", help="Skip DuckDuckGo web search")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    pipeline = CassavaPipeline()

    if args.classify_only:
        result = pipeline.classify(args.image)
    else:
        result = pipeline.run(args.image, question=args.question, search=not args.no_search)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "="*60)
        print(f"Disease : {result.get('display') or result.get('disease')}")
        print(f"Confidence: {result.get('confidence', 0)*100:.1f}%")
        print("\nTop predictions:")
        for name, prob in result.get("top5", []):
            bar = "#" * int(prob * 20)
            print(f"  {name:<25} {prob*100:5.1f}%  {bar}")
        if "advice" in result:
            print("\nAdvice for farmer:")
            print("-" * 60)
            print(result["advice"])
        if result.get("web_results"):
            print("\nWeb Resources:")
            print("-" * 60)
            for i, r in enumerate(result["web_results"], 1):
                print(f"{i}. {r['title']}")
                print(f"   {r['snippet'][:200]}")
                print(f"   {r['url']}")
        print("="*60)


if __name__ == "__main__":
    main()
