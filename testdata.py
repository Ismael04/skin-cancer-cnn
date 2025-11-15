from src.data import get_dataloaders

_, _, _, train_dl, val_dl, test_dl = get_dataloaders(
    data_dir="data",
    img_size=224,
    batch_size=4,   # petit batch pour tester
    num_workers=0   # 0 si ça bug sur Windows
)

images, labels = next(iter(train_dl))
print("Batch images:", images.shape)   # devrait afficher: torch.Size([4, 3, 224, 224])
print("Batch labels:", labels)         # 0=benign, 1=malignant (ordre dépend de l'alphabet)
