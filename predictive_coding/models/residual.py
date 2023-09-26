from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn


class Residual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1)
            self.skip = nn.Conv2d(in_channels, channels, kernel_size=1, stride=2)
        else:
            self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


class ResidualTranspose(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        upsample: bool = False
    ) -> None:
        super().__init__()
        self.upsample = upsample

        self.skip = None
        if in_channels < channels:
            self.skip = nn.ConvTranspose2d(in_channels, channels, kernel_size=1, stride=1)
        elif in_channels > channels:
            delta_channels = in_channels - channels
            self.skip = lambda x: torch.cat([x, torch.zeros(x.shape[0], delta_channels, x.shape[2], x.shape[3], device=x.device)], dim=1)

        if upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.skip = nn.ConvTranspose2d(in_channels, channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


class DownBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int,
        downsample: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Residual(in_channels, out_channels, downsample=downsample)])
        for _ in range(layers - 1):
            self.layers.append(Residual(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class UpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int,
        upsample: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ResidualTranspose(in_channels, out_channels, upsample=upsample)])
        for _ in range(layers - 1):
            self.layers.append(ResidualTranspose(out_channels, out_channels))

    def forward(self, x: torch.Tensor, skip: Union[torch.Tensor, None]) -> torch.Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class ConvBottleneck(nn.Module):
    def __init__(self, actions: int, in_channels: int, out_channels: int, features: int=4):
        super().__init__()
        self.features = features
        # self.linear = nn.Linear(actions, self.features * 8 * 8, bias=False)
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