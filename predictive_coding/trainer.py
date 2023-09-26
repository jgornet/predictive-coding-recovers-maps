from typing import Union
import os

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from PIL import Image
from torchvision.io import write_video
from tqdm.autonotebook import tqdm

from .models.models import Autoencoder, PredictiveCoder

class Trainer:
    def __init__(
            self, 
            model: Union[Autoencoder, PredictiveCoder],
            optimizer: Optimizer,
            scheduler,
            train_dataset: DataLoader,
            val_dataset: DataLoader, 
            checkpoint_path: str
        ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.scheduler = scheduler
        self.iteration = 1
        self.epoch = 1

        if not isinstance(self.model, PredictiveCoder) and not isinstance(self.model, Autoencoder):
            raise NotImplementedError("training is only implemented for Autoencoder or PredictiveCoder")

        self.mean = np.array([121.6697, 149.3242, 154.9510], dtype=np.float32)
        self.std = np.array([40.7521,  47.7267, 103.2739], dtype=np.float32)

    def fit(self, num_epochs):
        min_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        for epoch in tqdm(range(1, num_epochs + 1)):
            self.epoch = epoch

            train_loss = self.train()
            history['train_loss'].append(train_loss)

            val_loss = self.validate()
            history['val_loss'].append(val_loss)

            if val_loss < min_loss:
                min_loss = val_loss
                fn = 'best.ckpt'
                self.save_checkpoint(fn)

            fn = os.path.join(self.checkpoint_path, 'train_loss.npy')
            np.save(fn, np.array(history['train_loss']))
            fn = os.path.join(self.checkpoint_path, 'val_loss.npy')
            np.save(fn, np.array(history['val_loss']))

        return history

    def train(self, device='cuda:0'):
        self.model.train()
        with tqdm(self.train_dataset) as t:
            for batch_idx, batch in enumerate(t):
                t.set_description('BATCH {}'.format(batch_idx))
                inputs, actions, _ = batch
                inputs = inputs.to(device)
                actions = actions.to(device)

                self.optimizer.zero_grad()
                B, L, C, H, W = inputs.shape
                if isinstance(self.model, Autoencoder):
                    inputs = inputs.reshape(B*L, C, H, W)
                    actions = actions.reshape(B*L, 2)
                    predict = self.model(inputs, torch.zeros_like(actions))
                    loss = F.mse_loss(predict, inputs)
                if isinstance(self.model, PredictiveCoder):
                    predict = self.model(inputs, torch.zeros_like(actions))
                    loss = F.mse_loss(predict[:, :-1], inputs[:, 1:])
                else:
                    raise NotImplementedError("training is only implemented for Autoencoder or PredictiveCoder")
                loss = F.mse_loss(predict[:, :-1], inputs[:, 1:])
                t.set_postfix(loss=loss.item())

                t.update()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        return loss.item()

    def validate(self, device='cuda:0'):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataset):
                inputs, actions, _ = batch
                inputs = inputs.to(device)
                actions = actions.to(device)
                
                prediction = self.model(inputs, torch.zeros_like(actions))
                loss += F.mse_loss(prediction[:, :-1], inputs[:, 1:])

                if batch_idx > num_batch:
                    break

            loss /= num_batch
            tqdm.write('Test Loss: {:7f}'.format(loss.item()))

            inputs, prediction = inputs.reshape(-1, 3, 64, 64), prediction.reshape(-1, 3, 64, 64)

            save_path = os.path.join(self.checkpoint_path, 'epoch_{}.mp4'.format(self.epoch))
            tensor = torch.cat([inputs, prediction], dim=3)
            self.write_video(tensor, save_path)

        return loss.item()

    @torch.no_grad()
    def write_video(self, tensor, fn):
        array = self.std[None, None, None, :] * torch.permute(tensor, (0, 2, 3, 1)).cpu() + self.mean[None, None, None, :]
        array = torch.clamp(array, 0, 255)
        write_video(fn, array.type(torch.uint8), fps=10)

    def save_checkpoint(self, fn):
        save_path = os.path.join(self.checkpoint_path, fn)
        torch.save(self.model.state_dict(), save_path)