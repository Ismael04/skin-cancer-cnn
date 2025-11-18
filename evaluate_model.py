
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from src.data import get_dataloaders
from src.models import get_model
from src.utils import device

dev = device()
model = get_model("resnet50", num_classes=2, pretrained=False).to(dev)
model.load_state_dict(torch.load("model_resnet50.pth", map_location=dev))
model.eval()

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

cm = confusion_matrix(y_true, y_pred)
print("\nMatrice de confusion :\n", cm)

print("\nRapport de classification :\n", classification_report(y_true, y_pred, digits=3))

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print(f"AUC : {roc_auc:.3f}")

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - Détection du cancer de la peau ")
plt.legend()
plt.show()

j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = thresholds[best_idx]
print(f"Seuil optimal = {best_thresh:.3f}")

chosen_thresh = best_thresh
y_pred_new = (y_scores >= chosen_thresh).astype(int)

cm_new = confusion_matrix(y_true, y_pred_new)
print(f"\nMatrice de confusion (seuil {chosen_thresh:.3f}) :\n", cm_new)
print(f"\nRapport de classification (seuil {chosen_thresh:.3f}) :\n",
      classification_report(y_true, y_pred_new, digits=3))
