import os
from glob import glob
import argparse

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

from src.CustomUnetNN.segmentation_module import SegmentationModel
from configs.config_train_test import test_augmentation
from configs.config import class_rgb_values


def decode_outputs(outputs: torch.Tensor) -> torch.Tensor:
    """
    Decodes model outputs to RGB format.

    Args:
        outputs: Model output tensor.
    Returns:
        torch.Tensor: Decoded RGB image tensor.
    """
    rgb_image = torch.tensor(
        [
            list(list(class_rgb_values.keys())[elm.item()])
            for elm in outputs.argmax(axis=0).view(-1)
        ]
    )
    return rgb_image.permute(1, 0).view(3, outputs.shape[1], outputs.shape[2])


class ProductionModel:
    def __init__(self, transform: transforms, checkpoint_dir: str, data_dir: str):
        """
        Initializes the ProductionModel for inference.

        Args:
            transform: Image transformation function.
            checkpoint_dir: Path to the model checkpoint file.
            data_dir: Path to the directory containing test images.
        """
        self.model = self._load_model(checkpoint_dir, num_classes=7)
        self.image_paths = glob(os.path.join(data_dir, "*.jpg"))
        self.transform = transform
        self.model.eval()

    @staticmethod
    def _load_model(model_path: str, num_classes: int) -> torch.nn.Module:
        """
        Loads a model from a checkpoint file.

        Args:
            model_path: Path to the model checkpoint file.
            num_classes: Number of classes for the model.
        Returns:
            torch.nn.Module: Loaded model.
        """
        model = SegmentationModel(3, num_classes)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def predict(self, image_path: str, output_dir: str):
        """
        Performs inference on an image and saves the result.

        Args:
            image_path: Path to the input image.
            output_dir: Path to the directory to save the output image.
        """
        with torch.no_grad():
            input_tn = self.transform(Image.open(image_path))
            image_predict = decode_outputs(self.model(input_tn[None, :, :, :])[0])
            result_image = input_tn * image_predict

            plt.imshow(result_image.permute(2, 1, 0))
            plt.show()

            result_filename = os.path.basename(image_path)
            result_filepath = os.path.join(output_dir, result_filename)
            result_image_pil = transforms.ToPILImage()(result_image.cpu())
            result_image_pil.save(result_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Model Inference")

    parser.add_argument(
        "--checkpoint_dir",
        default="/home/anatoly/DeepGlobeClassificationPr/src/CustomUnetNN/checkpoints/best-checkpoint-v1.ckpt",
        type=str,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--data_dir",
        default="/home/anatoly/DeepGlobeClassificationPr/data/Data/test",
        type=str,
        help="Directory containing test images"
    )
    args = parser.parse_args()

    model_prod = ProductionModel(test_augmentation, args.checkpoint_dir, args.data_dir)

    output_directory = "Results"
    os.makedirs(output_directory, exist_ok=True)

    for image_path_predict in model_prod.image_paths:
        model_prod.predict(image_path_predict, output_directory)
