from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn

from .unet import UnetEncoder, UnetDecoder
from .attention import MultiHeadAttention


class Autoencoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = UnetEncoder(in_channels, layers)
        self.decoder = UnetDecoder(in_channels, out_channels, layers[::-1])
        self.bottleneck = ConvBottleneck(actions=3, in_channels=512, out_channels=512, features=128)

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


class PredictiveCoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int], seq_len: int, num_skip: int) -> None:
        super().__init__()
        self.encoder = UnetEncoder(in_channels, layers)
        self.decoder = UnetDecoder(in_channels, out_channels, layers[::-1])
        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.bottleneck = ConvBottleneck(actions=2, in_channels=512, out_channels=512, features=128)

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

class ConvBottleneck(nn.Module):
    def __init__(self, actions: int, in_channels: int, out_channels: int, features: int=4):
        super().__init__()
        self.features = features
        self.linear = nn.Linear(actions, self.features * 2 * 2, bias=False)
        self.conv = nn.Conv2d(self.features + in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, features, actions):
        B, C, H, W = features.shape
        conv_actions = self.linear(actions).reshape(B, self.features, H, W)

        output = torch.cat([features, conv_actions], dim=1)
        output = self.conv(output)
        output = self.bn(output)
        output = F.relu(output)

        return output