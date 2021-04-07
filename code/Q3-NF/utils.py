import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple
from solution import loss1


def density_arcs(z: torch.Tensor) -> torch.Tensor:
    """Return the probability of z for the density arc"""
    u = 0.5 * ((torch.norm(z, 2, dim=1) - 2) / 0.4) ** 2
    u1 = torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
    u2 = torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
    u = u - torch.log(u1 + u2)
    return u


def density_sine(z: torch.Tensor) -> torch.Tensor:
    """Return the probability of z for the density splitted sine"""
    w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
    u = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.4) ** 2)
    return -torch.log(u)


def plot_density(density, name: str = "density", xlim: float = 5, ylim: float = 5):
    """
    Plot the density and save the produced figure
    :param density: a function that return the probability
    :param str: the name of the figure to save
    :param xlim: limit used by the figure (-xlim, xlim)
    :param ylim: limit used by the figure (-ylim, ylim)
    """
    fig = plt.figure(figsize=(6, 6))
    x = np.linspace(-xlim, xlim, 100)
    y = np.linspace(-ylim, ylim, 100)
    x_, y_ = np.meshgrid(x, y)
    z = torch.from_numpy(np.concatenate([np.reshape(x_, (-1, 1)),
                                         np.reshape(y_, (-1, 1))], 1))
    u = torch.exp(-density(z))
    u = u.reshape(x_.shape)
    plt.pcolormesh(x_, y_, u, rasterized=True)

    fig.tight_layout()
    fig.savefig(f'{name}.png')


def plot_learned_density(model, name: str = "density", xlim: float = 5, ylim: float = 5):
    """
    Plot the density and save the produced figure
    :param density: a function that return the probability
    :param str: the name of the figure to save
    :param xlim: limit used by the figure (-xlim, xlim)
    :param ylim: limit used by the figure (-ylim, ylim)
    """
    fig = plt.figure(figsize=(6, 6))
    x = np.linspace(-xlim, xlim, 100)
    y = np.linspace(-ylim, ylim, 100)
    x_, y_ = np.meshgrid(x, y)
    z = torch.from_numpy(np.concatenate([np.reshape(x_, (-1, 1)),
                                         np.reshape(y_, (-1, 1))], 1))
    z = z.type(torch.FloatTensor)
    with torch.no_grad():
        x, logdet = model(z)
        u = torch.exp(-loss1(x, logdet))

    u = u.reshape(x_.shape).detach().numpy()
    plt.pcolormesh(x_, y_, u, rasterized=True)

    fig.tight_layout()
    fig.savefig(f'{name}.png')


def plot_samples(x: np.ndarray, y: np.ndarray, name: str = "samples", xlim: float = 5, ylim: float = 5):
    """
    Plot the samples x,y and save the produced figure
    :param x: array of value for x
    :param y: array of value for y
    :param str: the name of the figure to save
    :param xlim: limit used by the figure (-xlim, xlim)
    :param ylim: limit used by the figure (-ylim, ylim)
    """
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=3)
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    fig.tight_layout()
    fig.savefig(f'{name}.png')
