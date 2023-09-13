import os
from glob import glob
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from torchvision import transforms


train_augment = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)


class ImageDataset(Dataset):
    """

    PyTorch Dataset class for working with images and their masks.

    """

    def __init__(self, data_dir: str, transform: callable = None):
        """
        Initializes a new instance of the ImageDataset class.

        Args:
            data_dir: The path to the directory containing images and masks.
            transform: A transformation to apply to images and masks (default: None).
        """
        self.data_dir = data_dir
        self.dataset = pd.DataFrame(
            {
                "Images": sorted(glob(os.path.join(data_dir, "*.jpg"))),
                "Masks": sorted(glob(os.path.join(data_dir, "*.png"))),
            }
        )

        self.dataset = shuffle(self.dataset)
        self.dataset.reset_index(drop=True, inplace=True)

        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an image and its corresponding mask at index i.

        Args:
            i: The index of the item in the dataset.

        Returns:
            tuple: A tuple containing the image and the mask.
        """
        data_items = self.dataset.iloc[i]
        image = Image.open(data_items.Images)
        mask = Image.open(data_items.Masks)

        image = self.transform(image)
        mask = self.transform(mask).int()

        return image, mask
