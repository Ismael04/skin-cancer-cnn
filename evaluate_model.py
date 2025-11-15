# evaluate_model.py — évaluation complète du modèle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from src.data import get_dataloaders
from src.models import get_model
from src.utils import device

# === Chargement du modèle entraîné ===
dev = device()
model = get_model("resnet50", num_classes=2, pretrained=True).to(dev)
model.load_state_dict(torch.load("model.pth", map_location=dev))
model.eval()

# === Chargement des données de test ===
_, _, test_ds, _, _, test_dl = get_dataloaders(
    data_dir="data",
    img_size=128,
    batch_size=8,
    num_workers=0
)

y_true, y_pred, y_scores = [], [], []

with torch.no_grad():
    for x, y in test_dl:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        probs = torch.softmax(out, dim=1)[:, 1]  # proba classe 1 = cancer
        preds = torch.argmax(out, dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

y_true, y_pred, y_scores = np.array(y_true), np.array(y_pred), np.array(y_scores)

# === 1️⃣ Matrice de confusion ===
cm = confusion_matrix(y_true, y_pred)
print("\nMatrice de confusion :\n", cm)

# === 2️⃣ Rapport de classification ===
print("\nRapport de classification :\n", classification_report(y_true, y_pred, digits=3))

# === 3️⃣ Courbe ROC / AUC ===
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print(f"AUC : {roc_auc:.3f}")

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - Détection du cancer de la peau")
plt.legend()
plt.show()

# === 4️⃣ Seuil optimal ===
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = thresholds[best_idx]
print(f"Seuil optimal = {best_thresh:.3f}")
# === 5️⃣ Évaluation avec le seuil optimal ===
chosen_thresh = 0.29  # seuil trouvé par la courbe ROC

# Recalcul des prédictions avec ce seuil
y_pred_new = (y_scores >= chosen_thresh).astype(int)

# Nouvelle matrice de confusion et rapport
cm_new = confusion_matrix(y_true, y_pred_new)
print("\nMatrice de confusion (seuil 0.29) :\n", cm_new)
print("\nRapport de classification (seuil 0.29) :\n",
      classification_report(y_true, y_pred_new, digits=3))
