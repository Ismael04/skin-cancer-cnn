from src.data import get_dataloaders

_, _, _, train_dl, val_dl, test_dl = get_dataloaders(
    data_dir="data",
    img_size=224,
    batch_size=4,
    num_workers=0
)

images, labels = next(iter(train_dl))
print("Batch images:", images.shape)
print("Batch labels:", labels)
