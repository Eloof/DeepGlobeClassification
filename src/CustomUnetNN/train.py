import argparse
import mlflow
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dataLoader.data_module import DataModule

from src.CustomUnetNN.segmentation_module import SegmentationModel


class SegmentationTrainer:
    """
    A class for training a segmentation model.

    Args:
        arg: Command-line arguments and settings.
    """

    def __init__(self, arg: argparse.Namespace):
        """
        Initializes a new SegmentationTrainer instance.

        Args:
            arg: Command-line arguments and settings.
        """
        self.args = arg
        self.model = SegmentationModel(3, 7)
        self.trainer = None
        self.logger = None
        self.checkpoint_callback = None
        self.data_module = None
        self.device = None
        self.setup()

    def setup(self):
        """
        Set up the trainer, logger, data module, and device.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_module = DataModule(self.args)
        self.checkpoint_callback = self._create_checkpoint_callback()
        self.logger = self._create_logger()
        self.trainer = self._setup_trainer()

    @staticmethod
    def _create_checkpoint_callback() -> ModelCheckpoint:
        """
        Create a ModelCheckpoint callback for saving model checkpoints.

        Returns:
            ModelCheckpoint: ModelCheckpoint callback.
        """
        return ModelCheckpoint(
            dirpath="checkpoints", filename="best-checkpoint", save_top_k=1, mode="min"
        )

    @staticmethod
    def _create_logger() -> TensorBoardLogger:
        """
        Create a TensorBoardLogger for logging training information.

        Returns:
            TensorBoardLogger: TensorBoardLogger.
        """
        return TensorBoardLogger("lightning_logs", name="landcover-classification-log")

    def _setup_trainer(self) -> pl.Trainer:
        """
        Set up the PyTorch Lightning trainer for training the model.

        Returns:
            pl.Trainer: PyTorch Lightning Trainer.
        """
        return pl.Trainer(
            logger=self.logger,
            log_every_n_steps=10,
            callbacks=[self.checkpoint_callback],
            max_epochs=self.args.max_epochs,
            accelerator="gpu",
            devices=1,
            accumulate_grad_batches=2,
        )

    def run(self):
        """
        Run the training process.
        """
        with mlflow.start_run():
            mlflow.pytorch.autolog()
            self.trainer.fit(self.model, datamodule=self.data_module)
            mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Model Training")
    parser.add_argument(
        "--max_epochs", default=2, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--train_batch_size", default=16, type=int, help="Batch size"
    )
    parser.add_argument(
        "--val_batch_size", default=8, type=int, help="Validation batch size"
    )
    parser.add_argument(
        "--learning_rate", default=0.0005, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--num_workers", default=12, type=int, help="Number of data loader workers"
    )
    parser.add_argument(
        "--data_training_dir",
        default="data/Data/train",
        type=str,
        help="Directory for data loading",
    )
    args = parser.parse_args()

    trainer = SegmentationTrainer(args)
    trainer.run()
