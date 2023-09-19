import mlflow
import torch
import torchvision.transforms.functional as TF

from segmentation_models_pytorch.metrics import f1_score, get_stats, iou_score
from torch import nn

SEED = 42
torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")
mlflow.set_tracking_uri("http://127.0.0.1:5000")


class DoubleConv(nn.Module):
    """
    DoubleConv module consists of two consecutive convolutional layers followed by
    batch normalization and ReLU activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolutional kernel size.
        stride: Convolution stride.
        padding: Padding for convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DoubleConv module.

        Args:
            inputs: Input data.
        Returns:
            torch.Tensor: Output data after applying DoubleConv.
        """
        return self.conv(inputs)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_segmentations: int,
        features: tuple = (16, 32, 64, 128),
    ):
        """
        UNet model for image segmentation.

        Args:
            in_channels: Number of input channels.
            num_segmentations: Number of segmentations (classes).
            features: tuple of the number of filters in UNet layers.
        """
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss()

        in_channels_down_iter = in_channels
        self.down_part = nn.ModuleList()
        for feature in features:
            conv = DoubleConv(
                in_channels=in_channels_down_iter,
                out_channels=feature,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.down_part.append(conv)
            in_channels_down_iter = feature

        self.bottleneck = DoubleConv(
            in_channels=features[-1],
            out_channels=features[-1] * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_part = nn.ModuleList()
        for feature in reversed(features):
            conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                DoubleConv(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
            )
            self.up_part.append(conv)

        self.output = nn.Conv2d(
            in_channels=features[0], out_channels=num_segmentations, kernel_size=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        """
        Forward pass through the UNet model.

        Args:
            inputs: Input data.
            targets: Target segmentations.
        Returns:
            tuple: A tuple containing the loss, metrics, and model outputs.
        """
        skip_connections = []

        for down_conv in self.down_part:
            inputs = down_conv(inputs)
            skip_connections.append(inputs)
            inputs = self.pool(inputs)

        inputs = self.bottleneck(inputs)
        skip_connections = skip_connections[::-1]

        for i, up_conv in enumerate(self.up_part):
            inputs = up_conv[0](inputs)
            skip_connection = skip_connections[i]

            if inputs.shape != skip_connection.shape:
                inputs = TF.resize(inputs, size=skip_connection.shape[2:])
            concat_x = torch.cat((skip_connection, inputs), dim=1)

            inputs = up_conv[1](concat_x)

        outputs = self.output(inputs)

        if targets:
            loss = self.loss_func(outputs, targets)
            tp_val, fp_val, fn_val, tn_val = get_stats(
                outputs.argmax(dim=1).unsqueeze(1).type(torch.int64),
                targets.argmax(dim=1).unsqueeze(1).type(torch.int64),
                mode="multiclass",
                num_classes=7,
            )
            metrics = {
                "IoU": iou_score(
                    tp_val, fp_val, fn_val, tn_val, reduction="micro-imagewise"
                ),
                "F1score": f1_score(
                    tp_val, fp_val, fn_val, tn_val, reduction="micro-imagewise"
                ),
            }

            return loss, metrics, outputs

        return outputs
