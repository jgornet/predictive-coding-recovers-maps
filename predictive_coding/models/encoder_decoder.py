from typing import List, Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .residual import DownBlock2d, UpBlock2d


class Encoder(nn.Module):
    def __init__(self, in_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down1 = DownBlock2d(64, 64, layers[0])
        self.down2 = DownBlock2d(64, 128, layers[1], downsample=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x) # 32
        x = self.bn1(x)
        x = F.relu(x)

        x = self.maxpool(x) # 16

        x = self.down1(x) # 16
        x = self.down2(x) # 8

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(64, out_channels, kernel_size=1, stride=1)
        self.up1 = UpBlock2d(128, 64, layers[0], upsample=True)
        self.up2 = UpBlock2d(64, 64, layers[1], upsample=True)
        self.up3 = UpBlock2d(64, 64, 1, upsample=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x, None)
        x = self.up2(x, None)
        x = self.up3(x, None)

        x = self.conv1(x)

        return x