from torch.utils.data import Dataset
import pandas as pd
from sklearn.utils import shuffle
from glob import glob
from configs.config import HOME_DIR
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

train_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = pd.DataFrame(
            {
                "Images": sorted(glob(os.path.join(data_dir, "*.jpg"))),
                "Masks": sorted(glob(os.path.join(data_dir, "*.png"))),
            }
        )
        self.dataset = shuffle(self.dataset)
        self.dataset.reset_index(drop=True, inplace=True)

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data_items = self.dataset.iloc[i]

        image = Image.open(data_items.Images)
        mask = Image.open(data_items.Masks)

        if transforms:
            image = self.transform(image)
            mask = self.transform(mask)

        image = image.permute(1, 2, 0).clamp(0, 1).numpy()


if __name__ == "__main__":
    a = ImageDataset(os.path.join(HOME_DIR, "DeepGlobeClassification/data/Data/train"), train_augment)
    a[1]

