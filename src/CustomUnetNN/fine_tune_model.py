import argparse
import torch

from src.CustomUnetNN.train import SegmentationTrainer


class FineTuneModel(SegmentationTrainer):
    def __init__(self, arg):
        self.args = arg
        self.checkpoint = torch.load(self.args.checkpoint_dir)
        super().__init__(arg)
        self.model.load_state_dict(self.checkpoint["state_dict"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Model Training")
    parser.add_argument(
        "--max_epochs", default=1, type=int, help="Number of training epochs"
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
        default="DeepGlobeClassificationPr/data/Data/train",
        type=str,
        help="Directory for data loading",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/home/anatoly/DeepGlobeClassificationPr/src/CustomUnetNN/checkpoints/best-checkpoint-v2.ckpt",
        type=str,
        help="Directory checkpoint",
    )
    args = parser.parse_args()

    fine_tune_model = FineTuneModel(args)
    fine_tune_model.run()
