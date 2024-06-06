from typing import List, Dict, Union, Callable
from dataclasses import dataclass
from abc import ABC
from functools import cached_property
from os.path import join
from os import makedirs
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from tqdm.auto import tqdm

from .models.encoder_decoder import Autoencoder, PredictiveCoder

Color = np.ndarray
DistFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def circular_distance(pos0, pos1):
    dx = pos1[:, 0] - pos0[:, 0]
    ang = lambda y: np.mod(y, 30) / 30 * 2 * np.pi
    dang = np.mod(ang(pos1[:, 1]) - ang(pos0[:, 1]), 2 * np.pi)
    dy = np.zeros_like(dang)
    mask = dang <= np.pi
    dy[mask] = dang[mask] * 30 / (2 * np.pi)
    dy[~mask] = (2 * np.pi - dang[~mask]) * 30 / (2 * np.pi)

    return np.sqrt(dx**2 + dy**2)


def euclidean_distance(pos0, pos1):
    return np.sqrt(np.sum((pos0 - pos1) ** 2, axis=1))


@dataclass
class Latents:
    latents: np.ndarray
    positions: np.ndarray
    latent_distance: DistFn = euclidean_distance
    position_distance: DistFn = euclidean_distance
    L: int = 10000
    offset: int = 100
    neighborhood: int = 90

    @cached_property
    def distances(self) -> Dict[str, np.ndarray]:
        position_distances = self._dist(self.positions, self.position_distance)
        idx = np.argsort(position_distances)
        return {
            "latent": self.normalize(
                self._dist(
                    self.latents.reshape(len(self.latents), -1), self.latent_distance
                )
            )[idx],
            "position": position_distances[idx],
        }

    @staticmethod
    def normalize(array: np.ndarray) -> np.ndarray:
        vmax = array.max()
        vmin = array.min()
        return (array - vmin) / (vmax - vmin)

    def _dist(
        self,
        array: np.ndarray,
        distance_fn: DistFn,
    ):
        dist = np.zeros((self.L + 1) * self.L // 2)
        for shift in np.arange(1, self.L // 2):
            dist[(shift - 1) * self.neighborhood : shift * self.neighborhood] = (
                distance_fn(
                    array[self.offset : self.offset + self.neighborhood],
                    np.roll(
                        array[self.offset : self.offset + self.neighborhood],
                        shift - self.neighborhood // 2,
                        axis=0,
                    ),
                )
            )
        dist = dist[dist != 0]

        return dist


@dataclass
class LogRegression:
    alpha: float
    beta: float
    latents: Latents
    label: str
    color: Color

    def train(self, latents: Latents):
        def loss(params):
            alpha, beta = params
            latent = latents.distances["latent"]
            position = latents.distances["position"]
            N = int(len(latent) * 1.0)
            prediction = alpha * np.log(position[:N]) + beta
            weight = 1 / position[:N]
            return np.mean(weight * (prediction - latent[:N]) ** 2)

        result = minimize(loss, [self.alpha, self.beta])
        self.alpha, self.beta = result.x

    def evaluate(self, actual_distances: np.ndarray) -> np.ndarray:
        prediction = self.alpha * np.log(actual_distances) + self.beta

        return prediction


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


@dataclass
class PositionDecoder:
    batch_size: int = 512

    def __init__(self, in_dim: int = 128, device: str = "cuda"):
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, padding=1),
            nn.MaxPool2d(2),
            Lambda(lambda x: x.reshape(-1, 256 * 4 * 4)),
            nn.Linear(256 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.net.to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 4000, 0.1)

    def train(self, latents: Latents):
        inputs = torch.from_numpy(latents.latents).to(self.device)
        positions = torch.from_numpy(latents.positions).to(self.device) / 30
        for _ in (pbar := tqdm(range(8000))):
            batch_idx = np.arange(0, len(inputs))
            np.random.shuffle(batch_idx)
            batch_idx = batch_idx[
                : len(batch_idx) // self.batch_size * self.batch_size
            ].reshape(-1, self.batch_size)
            for it, idx in enumerate(batch_idx):
                self.optimizer.zero_grad()
                batch = inputs[idx].to(self.device)
                pos = positions[idx, :2].to(self.device)
                pred = self.net(batch)
                loss = F.mse_loss(pred, pos)
                loss.backward()
                self.optimizer.step()
                if it % 100:
                    with torch.no_grad():
                        pred = self.net(inputs[:1000]) * 30
                        pbar.set_postfix(
                            {
                                "loss": F.mse_loss(
                                    pred, positions[:1000, :2].to(self.device)
                                ).item()
                            }
                        )
            self.lr_scheduler.step()

    @torch.no_grad()
    def error(self, latents: Latents) -> np.ndarray:
        inputs = torch.from_numpy(latents.latents).to(self.device)
        pred = self.net(inputs).cpu() * 30
        error = np.linalg.norm((latents.positions - pred.numpy()), axis=1)

        return error


@dataclass
class PlaceFields:
    latents: Latents

    @cached_property
    def histogram(self) -> np.ndarray:
        histogram = []

        for idx in range(128):
            quant = np.quantile(np.mean(self.latents.latents, axis=(2, 3))[:, idx], 0.9)
            units = self.latents.positions[
                np.mean(self.latents.latents, axis=(2, 3))[:, idx] > quant
            ]
            hist = plt.hist2d(
                units[:, 0],
                units[:, 1],
                bins=(41, 66),
                cmap="Blues",
                range=[[-22, 22], [-30, 36]],
            )
            histogram.append(hist[0])

        histogram = np.stack(histogram, axis=0)
        return histogram

    @cached_property
    def gaussian(self) -> np.ndarray:
        gaussian = []
        for idx in range(128):
            latents = self.latents.latents
            positions = self.latents.positions
            quant = np.quantile(np.mean(latents, axis=(2, 3))[:, idx], 0.9)
            units = positions[np.mean(latents, axis=(2, 3))[:, idx] > quant]
            cov = np.cov(units, rowvar=False)
            mu = units.mean(axis=0).reshape(-1, 1, 1)

            grid = np.mgrid[-22:22:0.1, -30:36:0.1]

            gauss = (
                1
                / (2 * np.pi)
                * np.linalg.det(cov) ** (-0.5)
                * np.exp(
                    -0.5
                    * np.einsum(
                        "ijk,ijk->jk",
                        (grid - mu),
                        np.einsum("ij,jkl", np.linalg.inv(cov), (grid - mu)),
                    )
                )
            )
            gaussian.append(gauss)
        gaussian = np.stack(gaussian, axis=0)

        return gaussian

    @cached_property
    def approx_areas(self) -> np.ndarray:
        approx_areas = []
        for idx in range(128):
            latents = self.latents.latents
            positions = self.latents.positions
            quant = np.quantile(np.mean(latents, axis=(2, 3))[:, idx], 0.9)
            units = positions[np.mean(latents, axis=(2, 3))[:, idx] > quant]
            cov = np.cov(units, rowvar=False)
            approx_areas += [np.multiply.reduce(np.sqrt(np.linalg.svd(cov)[1])) * np.pi]

        return np.array(approx_areas)


@dataclass
class PlaceFieldDecoder:
    fields: PlaceFields
    model: LinearRegression = LinearRegression()

    def fit(self) -> LinearRegression:
        threshold = 1
        _, idx = np.unique(
            self.fields.histogram.reshape(128, -1) >= threshold,
            axis=1,
            return_inverse=True,
        )

        histogram = self.fields.histogram > 0
        sample_idx = idx != 0
        X = np.mgrid[0:41, 0:66].reshape(2, -1).astype(np.int32)[:, sample_idx]
        overlap = histogram[:, X[0], X[1]]

        self.model = self.model.fit(overlap.T, X.T)
        return self.model

    @cached_property
    def sample_idx(self) -> np.ndarray:
        threshold = 1
        _, idx = np.unique(
            self.fields.histogram.reshape(128, -1) >= threshold,
            axis=1,
            return_inverse=True,
        )

        sample_idx = idx != 0

        return sample_idx

    @cached_property
    def shuffle_idx(self) -> np.ndarray:
        shuffle_idx = np.random.choice(
            self.sample_idx.sum(), size=self.sample_idx.sum()
        )

        return shuffle_idx

    @cached_property
    def predicted_vector(self) -> np.ndarray:
        X = np.mgrid[0:41, 0:66].reshape(2, -1).astype(np.int32)[:, self.sample_idx]

        histogram = self.fields.histogram > 0
        overlap = histogram[:, X[0], X[1]]
        pred = self.model.predict(overlap.T)
        pred_vec = pred.T - pred[self.shuffle_idx].T

        return pred_vec.T

    @property
    def actual_vector(self) -> np.ndarray:
        X = np.mgrid[0:41, 0:66].reshape(2, -1).astype(np.int32)[:, self.sample_idx]
        true_vec = X - X[:, self.shuffle_idx]

        return true_vec.T

    @cached_property
    def actual_direction(self) -> np.ndarray:
        x, y = self.actual_vector[:, 0], self.actual_vector[:, 1]
        return np.arctan2(y, x)

    @cached_property
    def predicted_direction(self) -> np.ndarray:
        x, y = self.predicted_vector[:, 0], self.predicted_vector[:, 1]
        pred_theta = np.arctan2(y, x)
        actual_theta = self.actual_direction
        mask = pred_theta < (actual_theta - np.pi)
        pred_theta[mask] += 2 * np.pi
        mask = pred_theta > (actual_theta + np.pi)
        pred_theta[mask] -= 2 * np.pi

        return pred_theta

    @cached_property
    def actual_distance(self) -> np.ndarray:
        return np.linalg.norm(self.actual_vector, axis=1)

    @cached_property
    def predicted_distance(self) -> np.ndarray:
        return np.linalg.norm(self.predicted_vector, axis=1)


def generate_latents(
    model: Union[PredictiveCoder, Autoencoder], images: torch.Tensor
) -> np.ndarray:
    latents = []
    bsz = 100
    for idx in range(len(images) // bsz + 1):
        batch = images[bsz * idx : bsz * (idx + 1)]
        if len(batch) == 0:
            break
        with torch.no_grad():
            if isinstance(model, Autoencoder):
                if len(batch.shape) > 4:
                    B, L, C, H, W = batch.shape
                    batch = batch.reshape(B * L, C, H, W)
            features = model.get_latents(batch)
            latents.append(features.cpu())
    latents = torch.cat(latents, dim=0).numpy()

    return latents


def distribution_plot(
    regressions: List[LogRegression], density: float = 0.15
) -> Figure:
    fig = plt.figure()

    for reg in regressions:
        distances = reg.latents.distances
        actual = distances["position"]
        latent = distances["latent"]
        size = int(len(actual) * density)
        samples = np.random.randint(len(actual), size=size)
        sns.kdeplot(
            x=actual[samples],
            y=latent[samples],
            shade=True,
            label=reg.label,
            alpha=0.5,
            color=reg.color,
        )
        plt.scatter(
            actual[samples][::10],
            latent[samples][::10],
            s=0.05,
            alpha=0.7,
            zorder=1,
            color=reg.color,
        )
    plt.xlabel("Actual distance (lattice units)")
    plt.ylabel("Latent distance (a.u.)")

    regression = regressions[0]
    distances = regression.latents.distances
    actual = distances["position"]
    vmin, vmax = actual.min(), actual.max()
    alpha, beta = regression.alpha, regression.beta
    plt.plot(
        np.linspace(vmin, vmax, 1000),
        alpha * np.log(np.linspace(vmin, vmax, 1000)) + beta,
        "k-",
    )

    return fig


def error_map(decoder: PositionDecoder, latents: Latents) -> Figure:
    fig = plt.figure()

    positions = latents.positions
    error = decoder.error(latents)
    plt.hexbin(
        -positions[:, 1],
        positions[:, 0],
        C=error,
        gridsize=27,
        cmap="inferno",
        vmin=0,
        vmax=28,
        reduce_C_function=np.mean,
    )
    plt.colorbar(label=r"Error ($\Vert x - \hat{x} \Vert$) (lattice units)")
    plt.xlabel("x-axis (lattice units)")
    plt.ylabel("y-axis (lattice units)")

    return fig


def error_histogram(
    errors: List[np.ndarray],
    labels: list[str],
    colors: List[Color],
) -> Figure:
    fig = plt.figure()

    for error, label, color in zip(errors, labels, colors):
        sns.histplot(error, kde=True, stat="density", label=label, color=color)

    plt.xlabel("Error ($\Vert x - \hat{x}(z) \Vert_{\ell_2}$) (lattice units)")
    plt.legend()

    return fig


def qq_plot(regressions: List[LogRegression]) -> Figure:
    fig = plt.figure()

    for reg in regressions:
        distances = reg.latents.distances
        alpha, beta = reg.alpha, reg.beta
        plt.plot(
            alpha * np.log(np.sort(distances["position"])) + beta,
            np.sort(distances["latent"]),
            label=reg.label,
            color=reg.color,
        )
    vmin = alpha * np.log(regressions[0].latents.distances["position"].min()) + beta
    vmax = alpha * np.log(regressions[0].latents.distances["position"].max()) + beta
    plt.plot([vmin, vmax], [vmin, vmax], "k--", label="Regression Model")
    plt.xlabel("Regression Model (a.u.)")
    plt.ylabel("Latent Distance (a.u.)")

    return fig


def regression_plot(regressions: List[LogRegression]) -> Figure:
    fig = plt.figure()

    for reg in regressions:
        distances = reg.latents.distances
        alpha, beta = reg.alpha, reg.beta
        sort = np.argsort(distances["position"])
        plt.scatter(
            distances["position"][sort],
            distances["latent"][sort],
            s=0.05,
            alpha=0.075,
            color=reg.color,
        )
        plt.plot(
            np.linspace(0.1, 25), alpha * np.log(np.linspace(0.1, 25)) + beta, "k--"
        )
    plt.xscale("log")
    plt.xlabel("Actual distance (lattice units)")
    plt.ylabel("Latent distance (a.u.)")

    return fig


def kl_divergence(regression: LogRegression, bins=10) -> float:
    distance = regression.latents.distances
    position = distance["position"]
    latent = distance["latent"]

    p_xy, xedges, yedges = np.histogram2d(position, latent, bins=bins, density=True)
    p_xy /= p_xy.sum()

    prediction = regression.prediction
    q_xy = np.histogram2d(position, prediction, bins=[xedges, yedges], density=True)[0]
    q_xy /= q_xy.sum()

    kl = q_xy * np.log2(q_xy) - np.log2(p_xy)
    kl = kl[np.isfinite(kl)].sum()

    return kl


def mutual_information(latents: List[Latents]) -> List[float]:
    def MI(l: Latents):
        position, latent = l.distances["position"], l.distances["latent"]
        p_xy = np.histogram2d(position[200000:], latent[200000:], bins=10)[0]
        p_xy /= p_xy.sum()
        p_x = p_xy.sum(axis=0)
        p_y = p_xy.sum(axis=1)
        marginal = p_y[:, None] @ p_x[None, :]
        p_idx = p_xy != 0
        return np.sum((p_xy * (np.log2(p_xy) - np.log2(marginal)))[p_idx])

    return map(MI, latents)


def fields_plot(fields: PlaceFields) -> Figure:
    histogram = fields.histogram
    gaussian = fields.gaussian

    fig, axes = plt.subplots(nrows=16, ncols=8)

    for idx in range(128):
        i, j = idx // 8, idx % 8
        ax = axes[i, j]

        dalpha = 0.9
        im = histogram[idx] > 0
        ax.imshow(im, cmap="Blues", alpha=im * dalpha, extent=[-30, 36, 22, -22])
        ax.imshow(gaussian[idx], cmap="Blues", alpha=0.6, extent=[-30, 36, 22, -22])
        ax.axis("off")

    return fig


def fields_histogram(fields: PlaceFields) -> Figure:
    fig = plt.figure()

    sns.displot(fields.approx_areas, kde=True, fill=True)
    plt.xlabel("Area (Gaussian approximation, lattice units)")

    return fig


def fields_per_position(fields: PlaceFields) -> Figure:
    fig = plt.figure()

    histogram = fields.histogram
    mask = histogram.sum(axis=0) != 0
    plt.bar(
        np.arange(mask.sum()),
        np.sort((histogram[:, mask] > 0).reshape(128, -1).sum(axis=0)),
        width=1,
    )
    plt.ylim([0, 128])
    plt.xticks([])
    plt.ylabel("Number of active \nlatent units")
    plt.xlabel("Environment block")

    return fig


def positions_per_field(fields: PlaceFields) -> Figure:
    fig = plt.figure()

    histogram = fields.histogram
    mask = histogram.sum(axis=(1, 2)) != 0
    L = mask.sum()
    plt.bar(
        np.arange(128),
        np.sort((histogram[mask, :] > 0).reshape(L, -1).sum(axis=1)),
        width=1,
    )
    plt.xticks([])
    plt.ylabel("Number of\nactive environment\nblocks")
    plt.xlabel("Latent unit")

    return fig


def distance_histogram(decoder: PlaceFieldDecoder) -> Figure:
    fig = plt.figure()

    error = np.abs(decoder.actual_distance - decoder.predicted_distance)
    sns.histplot(error, stat="density")
    plt.xlabel("Predicted distance error (lattice units)")

    return fig


def direction_histogram(decoder: PlaceFieldDecoder) -> Figure:
    fig = plt.figure()

    error = np.abs(decoder.actual_direction - decoder.predicted_direction) * 180 / np.pi
    sns.histplot(error, stat="density")
    plt.xlabel("Predicted direction error (degrees)")

    return fig


def distance_regression(decoder: PlaceFieldDecoder) -> Figure:
    fig = plt.figure()

    plt.scatter(decoder.actual_distance, decoder.predicted_distance, marker="*", s=0.05)
    plt.plot(np.linspace(0, 60), np.linspace(0, 60), color="k")
    plt.xlabel("Actual distance\n(lattice units)")
    plt.ylabel("Predicted distance\n(lattice units)")
    plt.gca().set_aspect("equal")

    return fig


def direction_regression(decoder: PlaceFieldDecoder) -> Figure:
    fig = plt.figure()

    plt.scatter(
        decoder.actual_direction * 180 / np.pi,
        decoder.predicted_direction * 180 / np.pi,
        marker="*",
        s=0.05,
    )
    plt.plot(np.linspace(-180, 180), np.linspace(-180, 180), color="k")
    plt.xticks([-180, 180], ["$-180\degree$", "$180\degree$"])
    plt.yticks([-180, 180], ["$-180\degree$", "$180\degree$"])
    plt.xlabel("Actual direction\n(degrees)")
    plt.ylabel("Predicted direction (degrees)")
    plt.gca().set_aspect("equal")

    return fig


def error_circle(decoder: PositionDecoder, latents: Latents) -> Figure:
    fig = plt.figure()

    positions = latents.positions
    error = decoder.error(latents)

    ang = np.mod(positions[:, 1], 30) * 2 * np.pi / 30
    r = positions[:, 0]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    norm = Normalize(vmin=0, vmax=8)
    c = plt.cm.inferno(norm(error))
    ax.scatter(ang, r, c=c, s=error)
    ax.set_rlim([-2.5, 2.5])
    ax.set_rorigin(-5)
    ax.set_rgrids([-2, -1, 0, 1, 2], ["", "", "", "", ""])
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315], ["", "", "", "", "", "", "", ""]
    )
    t = np.linspace(-2.5, 6 - 2.5, 100) / 30 * 2 * np.pi
    dt = 6 / 30 * 2 * np.pi
    ax.fill_between(t, -2.5 * np.ones(100), 2.5 * np.ones(100), color="red", alpha=0.1)
    ax.fill_between(
        t + dt, -2.5 * np.ones(100), 2.5 * np.ones(100), color="green", alpha=0.1
    )
    ax.fill_between(
        t + 2 * dt, -2.5 * np.ones(100), 2.5 * np.ones(100), color="red", alpha=0.1
    )
    ax.fill_between(
        t + 3 * dt, -2.5 * np.ones(100), 2.5 * np.ones(100), color="yellow", alpha=0.1
    )
    ax.fill_between(
        t + 4 * dt, -2.5 * np.ones(100), 2.5 * np.ones(100), color="blue", alpha=0.1
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.inferno),
        ax=ax,
        label="$\Vert x - \hat{x}(z) \Vert_{\ell_2}$",
    )

    return fig
