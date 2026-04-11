from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Model(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 19,
    ) -> None:
        super().__init__()
        self.net = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)