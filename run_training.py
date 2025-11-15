# run_training.py — version complète du ResNet50 (toutes les couches entraînées)
import torch, torch.nn as nn, torch.optim as optim
from src.data import get_dataloaders
from src.models import get_model
from src.utils import set_seed, device

cfg = {
    "seed": 42,
    "img_size": 128,     # taille d'image standard pour ResNet
    "batch_size": 4,     # pas trop gros pour éviter de surcharger le CPU
    "epochs": 1,         # commence petit, tu pourras augmenter ensuite
    "lr": 1e-4,          # plus petit learning rate (important ici)
    "weight_decay": 1e-4,
    "num_workers": 0     # Windows → toujours 0
}

set_seed(cfg["seed"])
dev = device()
print("Device:", dev)

# === Chargement des données ===
_, _, _, train_dl, val_dl, _ = get_dataloaders(
    data_dir="data",
    img_size=cfg["img_size"],
    batch_size=cfg["batch_size"],
    num_workers=cfg["num_workers"],
)

# === Modèle complet : ResNet50 pré-entraîné (toutes les couches apprennent) ===
model = get_model("resnet50", num_classes=2, pretrained=True).to(dev)

# === Optimiseur / fonction de perte ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

# === Entraînement complet ===
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

    print(f"[{epoch}] loss={running/total:.4f}  acc={correct/total:.3f}")

# === Validation ===
model.eval()
total, correct = 0, 0
with torch.no_grad():
    for x, y in val_dl:
        x, y = x.to(dev), y.to(dev)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Val acc: {correct/total:.3f}")
print("✅ Entraînement complet terminé.")

torch.save(model.state_dict(), "model.pth")
print("💾 Modèle sauvegardé sous 'model.pth'")


