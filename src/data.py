# src/data.py
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Transformations simples:
# - redimensionner à 224x224
# - convertir en tenseur
# - normaliser (valeurs standard ImageNet, ça marche bien avec les CNN)
def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),        # un peu d'augmentation
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

def get_dataloaders(data_dir="data", img_size=224, batch_size=16, num_workers=2):
    """
    Attend une structure:
    data/
      train/benign|malignant
      val/benign|malignant
      test/benign|malignant
    """
    train_tf, val_tf = get_transforms(img_size)

    train_ds = ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_ds   = ImageFolder(f"{data_dir}/val",   transform=val_tf)
    test_ds  = ImageFolder(f"{data_dir}/test",  transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl
