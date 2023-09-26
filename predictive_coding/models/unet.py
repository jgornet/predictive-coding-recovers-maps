from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn

from .residual import DownBlock2d, UpBlock2d
from .attention import MultiHeadAttention


class UnetEncoder(nn.Module):
    def __init__(self, in_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down1 = DownBlock2d(64, 64, layers[0])
        self.down2 = DownBlock2d(64, 128, layers[1], downsample=True)
        self.down3 = DownBlock2d(128, 256, layers[2], downsample=True)
        self.down4 = DownBlock2d(256, 512, layers[3], downsample=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.conv1(x) # 32
        x = self.bn1(x)
        x = F.relu(x)

        x = self.maxpool(x) # 16

        x = self.down1(x) # 16
        features.append(self.down2(x)) # 8
        features.append(self.down3(features[-1])) # 4
        features.append(self.down4(features[-1])) # 2

        return features


class UnetDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(64, out_channels, kernel_size=1, stride=1)
        self.up1 = UpBlock2d(512, 256, layers[0], upsample=True)
        self.up2 = UpBlock2d(512, 128, layers[1], upsample=True)
        self.up3 = UpBlock2d(256, 64, layers[2], upsample=True)
        self.up4 = UpBlock2d(64, 64, layers[3], upsample=True)
        self.up5 = UpBlock2d(64, 64, 1, upsample=True)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = self.up1(features[-1], None)
        x = self.up2(x, features[-2])
        x = self.up3(x, features[-3])
        x = self.up4(x, None)

        x = self.up5(x, None)

        x = self.conv1(x)

        return x

    def get_codes(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        codes = []
        x = self.up1(features[-1], None)
        codes.append(x)
        x = self.up2(x, features[-2])
        codes.append(x)
        x = self.up3(x, features[-3])
        codes.append(x)
        x = self.up4(x, None)
        codes.append(x)

        x = self.up5(x, None)
        codes.append(x)

        return codes


class Autoencoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = UnetEncoder(in_channels, layers)
        self.bottleneck = ConvBottleneck(actions=3, in_channels=512, out_channels=512, features=128)
        self.decoder = UnetDecoder(in_channels, out_channels, layers[::-1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        x = self.decoder(features)

        return x


class Unet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int], seq_len: int, num_skip: int) -> None:
        super().__init__()
        self.encoder = UnetEncoder(in_channels, layers)
        self.bottleneck = ConvBottleneck(actions=2, in_channels=512, out_channels=512, features=128)
        self.decoder = UnetDecoder(in_channels, out_channels, layers[::-1])
        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        self.attention = nn.ModuleList([
            MultiHeadAttention(128),
            MultiHeadAttention(256),
            MultiHeadAttention(512)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x = x.reshape(B*L, C, H, W)

        action = action.reshape(B*L, -1)
        features = self.encoder(x)
        features[-1] = self.bottleneck(features[-1], action)

        features = [f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3]) for f in features]
        features = [attention(feature, feature, feature, self.mask)[0] for attention, feature in zip(self.attention, features)]
        features = [f.reshape(B*L, f.shape[2], f.shape[3], f.shape[4]) for f in features]

        x = self.decoder(features)
        x = x.reshape(B, L, C, H, W)

        return x
