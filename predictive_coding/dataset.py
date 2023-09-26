from typing import List, Tuple
from pathlib import Path

from glob import glob
from os.path import join

import numpy as np
import torch
import torch.optim
from torchvision import transforms
from PIL import Image

from torch.utils.data import Dataset


class EnvironmentDataset(Dataset):
    def __init__(self, root: Path, sequence_length=10):
        self.episodes = list(root.glob("*"))
        self.episodes.sort()

        episode_length = len(list(self.episodes[0].glob('*')))
        self.sequence_length = sequence_length
        self.seq_per_epi = episode_length // sequence_length

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [121.6697/255, 149.3242/255, 154.9510/255], 
                [40.7521/255,  47.7267/255, 103.2739/255]
            )
        ])

    def __len__(self):
        return len(self.episodes * self.seq_per_epi)

    def __getitem__(self, idx):
        episode = self.episodes[idx // (self.seq_per_epi)]
        fns = glob(join(episode, "*.png"))
        fns.sort()

        images = [self.transforms(Image.open(fn)) for fn in fns]
        images = torch.stack(images, dim=0)

        actions = torch.from_numpy(
            np.load(join(episode, "actions.npz"))["arr_0"]
        )

        actions[:, 0] = (actions[:, 0] - 0.135) / 0.27
        actions[:, 1] = actions[:, 1] / 0.00378749
        
        state = torch.from_numpy(
            np.load(join(episode, "state.npz"))["arr_0"]
        )
        
        offset = idx % (self.seq_per_epi)
        L = self.sequence_length
        return images[L*offset:L*(offset+1)], actions[L*offset:L*(offset+1)], state[L*offset:L*(offset+1)]

def collate_fn(batch: List[Tuple[torch.Tensor]]):
    images = [item[0] for item in batch]
    images = torch.stack(images, dim=0)
    images = images.to(memory_format=torch.contiguous_format).float()

    actions = [item[1] for item in batch]
    actions = torch.stack(actions, dim=0)
    actions = actions.to(memory_format=torch.contiguous_format).float()
    
    state = [item[2] for item in batch]
    state = torch.stack(state, dim=0)
    state = state.to(memory_format=torch.contiguous_format).float()

    return images, actions, state