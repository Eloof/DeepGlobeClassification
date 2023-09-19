import argparse
import os
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from configs.config import HOME_DIR
from configs.config_train_test import train_augmentation

from dataLoader.data_loader import ImageDataset


class DataModule(pl.LightningDataModule):
    """
    LightningDataModule for handling data loading and splitting for training and validation.

    Args:
        args: Command-line arguments and settings.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initializes a new DataModule instance.

        Args:
            args: Command-line arguments and settings.
        """
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.args = args

    def setup(self, stage=None):
        """
        Set up the training and validation datasets.

        Args:
            stage (str, optional): One of 'fit' or 'test'. Defaults to None.
        """
        dataset = ImageDataset(
            os.path.join(HOME_DIR, self.args.data_training_dir),
            transform=train_augmentation,
        )
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )
