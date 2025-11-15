import os, shutil, pandas as pd

# chemins
csv_path = "data/HAM10000_metadata.csv"      # fichier CSV
img_dir = "data/raw_images"                  # dossier avec toutes les images
output_dir = "data"                          # dossier de sortie

# lecture du fichier CSV
df = pd.read_csv(csv_path)

# classes (malignes vs bénignes)
malignant = ["mel", "bcc", "akiec"]
benign = ["bkl", "df", "nv", "vasc"]

# création des dossiers de sortie
for subset in ["train", "val", "test"]:
    for cls in ["benign", "malignant"]:
        os.makedirs(os.path.join(output_dir, subset, cls), exist_ok=True)

# mélange des données
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# séparation train/val/test
n = len(df)
train_df = df[:int(0.8*n)]
val_df   = df[int(0.8*n):int(0.9*n)]
test_df  = df[int(0.9*n):]

# fonction pour copier les images
def copy_images(subset_df, subset_name):
    for _, row in subset_df.iterrows():
        src = os.path.join(img_dir, f"{row['image_id']}.jpg")
        if not os.path.exists(src):
            continue
        cls = "malignant" if row["dx"] in malignant else "benign"
        dst = os.path.join(output_dir, subset_name, cls, f"{row['image_id']}.jpg")
        shutil.copy(src, dst)

print("Copie des images (train)...")
copy_images(train_df, "train")
print("Copie des images (val)...")
copy_images(val_df, "val")
print("Copie des images (test)...")
copy_images(test_df, "test")

print("✅ Dataset prêt dans le dossier 'data/' !")
