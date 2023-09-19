from torchvision import transforms

train_augmentation = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize(
            size=(265, 265), interpolation=transforms.InterpolationMode.NEAREST
        ),
    ]
)

test_augmentation = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            size=(265, 265), interpolation=transforms.InterpolationMode.NEAREST
        )
    ]
)
