"""
EfficientNet-B4 fine-tuning for cassava leaf disease classification.
GPU-optimized for Google Colab (T4/A100/V100). Falls back to CPU.
Target: ~85% val accuracy on GPU (vs 66% CPU MobileNetV2 baseline).
Classes: bacterial_blight, brown_streak_disease, green_mottle, healthy, mosaic_disease
"""

import os
import time
import json
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
NUM_WORKERS = 0 if platform.system() == "Windows" else 4

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/cassava-dataset/data"
MODEL_SAVE_PATH = "models/cassava_classifier.pth"
RESULTS_PATH = "data/image_classifier_results.json"

IMG_SIZE = 380          # EfficientNet-B4 native resolution
EPOCHS = 25
LR_HEAD = 1e-3          # higher LR for new classifier head
LR_BACKBONE = 5e-5      # low LR for pre-trained backbone
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
BATCH_SIZE = 64 if DEVICE.type == "cuda" else 32

CLASS_NAMES = [
    "Cassava___bacterial_blight",
    "Cassava___brown_streak_disease",
    "Cassava___green_mottle",
    "Cassava___healthy",
    "Cassava___mosaic_disease",
]
DISEASE_MAP = {
    "Cassava___bacterial_blight": "CBB",
    "Cassava___brown_streak_disease": "CBSD",
    "Cassava___green_mottle": "CGM",
    "Cassava___healthy": "Healthy",
    "Cassava___mosaic_disease": "CMD",
}

# ── Transforms ────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def make_train_sampler(targets):
    """Weighted sampler to counter class imbalance (mosaic disease = 62%)."""
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[t] for t in targets], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if USE_AMP:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels


def main():
    print(f"Device: {DEVICE} | AMP: {USE_AMP} | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
    print(f"Model: EfficientNet-B4 | Input: {IMG_SIZE}x{IMG_SIZE} | Epochs: {EPOCHS}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")

    # ── Dataset split ─────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_base = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    val_base = datasets.ImageFolder(DATA_DIR, transform=val_transforms)
    class_names = train_base.classes
    n = len(train_base)
    print(f"Classes: {class_names}")
    print(f"Total images: {n}")

    val_size = int(0.2 * n)
    train_size = n - val_size
    shuffled = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx, val_idx = shuffled[:train_size], shuffled[train_size:]

    train_dataset = Subset(train_base, train_idx)
    val_dataset = Subset(val_base, val_idx)

    train_targets = [train_base.targets[i] for i in train_idx]
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=make_train_sampler(train_targets),
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
    )
    print(f"Train: {train_size} | Val: {val_size}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Differential LR: backbone gets 20x lower LR than head
    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler() if USE_AMP else None

    os.makedirs("models", exist_ok=True)
    best_val_acc = 0.0
    history = []

    print(f"\nTraining for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_arch": "efficientnet_b4",
                "img_size": IMG_SIZE,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
            }, MODEL_SAVE_PATH)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n== Final Evaluation ==")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_acc, final_preds, final_labels = eval_epoch(model, val_loader, criterion)

    short_names = [DISEASE_MAP.get(c, c) for c in class_names]
    report = classification_report(final_labels, final_preds, target_names=short_names, output_dict=True)
    print(classification_report(final_labels, final_preds, target_names=short_names))

    cm = confusion_matrix(final_labels, final_preds).tolist()
    results = {
        "model_arch": "efficientnet_b4",
        "best_val_acc": round(best_val_acc, 4),
        "final_val_acc": round(final_acc, 4),
        "epochs_trained": EPOCHS,
        "img_size": IMG_SIZE,
        "device": str(DEVICE),
        "class_names": class_names,
        "disease_map": DISEASE_MAP,
        "history": history,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest val accuracy : {best_val_acc:.4f}")
    print(f"Model saved       : {MODEL_SAVE_PATH}")
    print(f"Results saved     : {RESULTS_PATH}")


if __name__ == "__main__":
    main()
