import torch

import pytorch_lightning as pl
from src.CustomUnetNN.unet_nn import UNet


class SegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for segmentation tasks.

    Args:
        in_channels: Number of input channels.
        num_segmentations: Number of segmentation classes.
    """

    def __init__(self, in_channels: int, num_segmentations: int):
        super().__init__()
        self.model = UNet(in_channels, num_segmentations)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        """
        Forward pass of the segmentation model.

        Args:
            x: Input data.
            targets: Ground truth targets.
        Returns:
            tuple: A tuple containing the loss, metrics, and model outputs.
        """
        return self.model(x, targets)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step for the segmentation model.

        Args:
            batch: A tuple containing input data and targets.
            batch_idx: Index of the current batch.
        Returns:
            torch.Tensor: The training loss.
        """
        input_tn, mask = batch
        loss, metrics, outputs = self(input_tn, mask)

        self.log_dict(
            {
                "train/Loss": loss,
                "train/IoU": metrics["IoU"],
                "train/F1score": metrics["F1score"],
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the segmentation model.

        Args:
            batch: A tuple containing input data and targets.
            batch_idx: Index of the current batch.
        """
        input_tn, mask = batch
        loss, metrics, outputs = self(input_tn, mask)
        self.log_dict(
            {
                "val/Loss": loss,
                "val/IoU": metrics["IoU"],
                "val/F1score": metrics["F1score"],
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        """
        Configure the optimizer for training the model.

        Returns:
            torch.optim.Optimizer: The optimizer for training the model.
        """
        optimizer = torch.optim.NAdam(self.parameters(), lr=0.0005)
        return optimizer
