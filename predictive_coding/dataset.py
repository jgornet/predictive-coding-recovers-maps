from typing import List, Tuple
from pathlib import Path
from dataclasses import dataclass
from glob import glob
from os.path import join

import numpy as np
import torch
import torch.optim
from torchvision import transforms
from PIL import Image

from torch.utils.data import Dataset
import torch


class EnvironmentDataset(Dataset):
    def __init__(self, root: Path, sequence_length=10):
        self.episodes = list(root.glob("*"))
        self.episodes.sort()

        episode_length = len(list(self.episodes[0].glob("*")))
        self.sequence_length = sequence_length
        self.seq_per_epi = episode_length // sequence_length

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [121.6697 / 255, 149.3242 / 255, 154.9510 / 255],
                    [40.7521 / 255, 47.7267 / 255, 103.2739 / 255],
                ),
            ]
        )

    def __len__(self):
        return len(self.episodes * self.seq_per_epi)

    def __getitem__(self, idx):
        episode = self.episodes[idx // (self.seq_per_epi)]
        fns = glob(join(episode, "*.png"))
        fns.sort()

        images = [self.transforms(Image.open(fn)) for fn in fns]
        images = torch.stack(images, dim=0)

        state = torch.from_numpy(np.load(join(episode, "state.npz"))["arr_0"])

        offset = idx % (self.seq_per_epi)
        L = self.sequence_length
        return (
            images[L * offset : L * (offset + 1)],
            state[L * offset : L * (offset + 1)],
        )


class CircleDataset(Dataset):
    def __init__(self, images_fn, positions_fn, length=50, speed=1):
        self.inputs = torch.from_numpy(np.load(images_fn))
        H, W, C = self.inputs.shape[1:]

        # Truncate out modulus of sequence length
        self.inputs = self.inputs[
            : (len(self.inputs) // (length * speed)) * (length * speed)
        ]

        # Normalize inputs
        self.inputs = (self.inputs / 128) - 1.0

        # Reshape for sequence length
        self.inputs = self.inputs.permute((0, 3, 1, 2))
        self.inputs = self.inputs.reshape(-1, length, speed, C, H, W)
        self.inputs = self.inputs.permute((0, 2, 1, 3, 4, 5))
        self.inputs = self.inputs.reshape(-1, length, C, H, W)

        self.positions = torch.from_numpy(np.load(positions_fn))

        # Truncate out modulus of sequence length
        self.positions = self.positions[
            : (len(self.positions) // (length * speed)) * (length * speed)
        ]

        # Reshape for sequence length
        self.positions = self.positions.reshape(-1, length, speed, 3)
        self.positions = self.positions.permute((0, 2, 1, 3))
        self.positions = self.positions.reshape(-1, length, 3)

        self.speed = speed

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        _idx = (idx * self.speed) % len(self.inputs) + idx // len(self.inputs)
        return (self.inputs[_idx], self.positions[_idx])


@dataclass
class TensorDataset(Dataset):
    images: torch.Tensor
    positions: torch.Tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.positions[idx]


def collate_fn(batch):
    images = torch.stack([sample[0] for sample in batch], dim=0)
    positions = torch.stack([sample[1] for sample in batch], dim=0)
    return (
        images.to(memory_format=torch.contiguous_format).float(),
        positions.to(memory_format=torch.contiguous_format).float(),
    )
