import os
from glob import glob
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from configs.config import class_rgb_values


def mask_one_hot_encode(mask: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a mask tensor using class RGB values.

    Args:
        mask: The input mask tensor.
    Returns:
        torch.Tensor: The one-hot encoded mask tensor.

    """
    _, h, w = mask.shape
    mask_flat = mask.permute(1, 2, 0).int().view(-1, 3)
    torch_mask_enc = torch.tensor(
        [class_rgb_values[tuple(rgb.tolist())] for rgb in mask_flat]
    ).view(h, w)
    return (
        F.one_hot(torch_mask_enc, num_classes=len(class_rgb_values))
        .permute(2, 1, 0)
        .float()
    )


class ImageDataset(Dataset):
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

        self.dataset.reset_index(drop=True, inplace=True)

        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, indx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an image and its corresponding mask at index i.

        Args:
            indx: The index of the item in the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and the mask.
        """
        data_items = self.dataset.iloc[indx]

        image = self.transform(Image.open(data_items.Images))
        mask = mask_one_hot_encode(self.transform(Image.open(data_items.Masks)))

        return image, mask
