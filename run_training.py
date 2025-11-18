# run_training.py — Entraînement ResNet50 (finetuning complet)

import torch
import torch.nn as nn
import torch.optim as optim
from src.data import get_dataloaders
from src.models import get_model
from src.utils import set_seed, device

# ========= CONFIG =========
cfg = {
    "seed": 42,
    "img_size": 128,
    "batch_size": 8,
    "epochs": 3,           # même nombre que EfficientNet pour comparaison juste
    "lr": 1e-4,            # IMPORTANT pour le finetuning ResNet
    "weight_decay": 1e-4,
    "num_workers": 0
}

MODEL_NAME = "resnet50"
SAVE_PATH = f"model_{MODEL_NAME}.pth"

# ==========================

set_seed(cfg["seed"])
dev = device()
print("Device:", dev)
print("Modèle entraîné :", MODEL_NAME)
print("Fichier de sortie :", SAVE_PATH)

# === Data ===
_, _, _, train_dl, val_dl, _ = get_dataloaders(
    data_dir="data",
    img_size=cfg["img_size"],
    batch_size=cfg["batch_size"],
    num_workers=cfg["num_workers"],
)

# === Modèle (ResNet50 pré-entraîné) ===
model = get_model(MODEL_NAME, num_classes=2, pretrained=True).to(dev)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

# === Entraînement ===
for epoch in range(1, cfg["epochs"] + 1):
    model.train()
    total, correct, running = 0, 0, 0.0

    for x, y in train_dl:
        x, y = x.to(dev), y.to(dev)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    train_loss = running / total
    train_acc = correct / total
    print(f"[{epoch}] loss={train_loss:.4f}  acc={train_acc:.3f}")

    # validation
    model.eval()
    v_total, v_correct = 0, 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(dev), y.to(dev)
            preds = model(x).argmax(1)
            v_correct += (preds == y).sum().item()
            v_total += y.size(0)
    val_acc = v_correct / v_total
    print(f"    -> Val acc: {val_acc:.3f}")

# === Sauvegarde ===
torch.save(model.state_dict(), SAVE_PATH)
print(f"💾 Modèle sauvegardé sous '{SAVE_PATH}'")
print("✅ Finetuning ResNet50 terminé.")
