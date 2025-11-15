import os, random

# === Paramètres ===
base_dir = "data"
target_total = 4000  # nombre total d'images à garder
classes = ["benign", "malignant"]
random.seed(42)

# === Calcul du nombre par classe ===
target_per_class = target_total // len(classes)

# === Parcours de tous les sous-dossiers ===
kept, deleted = 0, 0

for subset in ["train", "val", "test"]:
    for cls in classes:
        folder = os.path.join(base_dir, subset, cls)
        if not os.path.exists(folder):
            continue

        images = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        n = len(images)
        if n > target_per_class:
            # Sélection aléatoire d'images à garder
            keep = set(random.sample(images, target_per_class))
            for img in images:
                if img not in keep:
                    try:
                        os.remove(img)
                        deleted += 1
                    except Exception as e:
                        print(f"Erreur en supprimant {img}: {e}")
            kept += target_per_class
        else:
            kept += n

print(f"✅ Réduction terminée !")
print(f"   Total conservé : {kept} images")
print(f"   Total supprimé : {deleted} images")
print(f"   Objectif : {target_total} images équilibrées (≈{target_per_class} par classe)")
