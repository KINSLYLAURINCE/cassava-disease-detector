"""Download models from Hugging Face Hub at build time."""
import os
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "kinslydebruyne17/cassava-disease-models"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Downloading EfficientNet model...")
hf_hub_download(
    repo_id=REPO_ID,
    filename="cassava_efficientnet_b4.pth",
    local_dir=MODELS_DIR,
)

print("Downloading TinyLlama base model...")
snapshot_download(
    repo_id=REPO_ID,
    allow_patterns="tinyllama-1.1b-chat/*",
    local_dir=MODELS_DIR,
)

print("Downloading TinyLlama fine-tuned adapter...")
snapshot_download(
    repo_id=REPO_ID,
    allow_patterns="tinyllama-cassava-finetuned/*",
    local_dir=MODELS_DIR,
)

print("All models downloaded.")
