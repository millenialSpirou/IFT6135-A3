# -*- coding: utf-8 -*-
r"""
:mod:`dirac` -- Implement components and visualization for dirac task
=====================================================================

.. module:: dirac
   :platform: Unix
   :synopsis: Modules for generator/critic of dirac game and render trajectories

"""
import subprocess
import torch
from torch import nn

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation, rc
import numpy as np
import colorcet as cc
import seaborn as sns


cmap = cc.cm.glasbey_cool
sns.set_theme(style="dark")
SMALL_SIZE = 18
MEDIUM_SIZE = 24
BIGGER_SIZE = 32
rc('font', size=SMALL_SIZE)
rc('figure', titlesize=SMALL_SIZE)
rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=14)    # fontsize of the tick labels
rc('ytick', labelsize=14)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
rc('animation', html='jshtml')


class DiracGenerator(nn.Module):
    def __init__(self, std=0.7):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1) * std)

    def forward(self, z=None):
        return self.param


class DiracCritic(nn.Module):
    def __init__(self, std=0.7, num_params=1):
        super().__init__()
        self.param = nn.Parameter(torch.randn(2) * std)

    def project_lipschitz(self, weight_clipping=1.):
        with torch.no_grad():
            self.param[:, 0].clamp_(-weight_clipping, weight_clipping)

    def forward(self, x):  # (N, )
        x = torch.stack([x, torch.ones_like(x)], dim=-1)  # (N, 2)
        return self.param.mul(x).sum(-1)  # (N, )


def animate(trajectory, hps):
    trajectory = trajectory[:, None, :]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r"$\psi_0$", labelpad=14)
    ax.set_ylabel(r"$\psi_1$", labelpad=14)
    ax.set_zlabel(r"$\theta$", labelpad=14)
    reg_info = ''
#     if hps.weight_clipping:
#         reg_info += '+lipschitz proj'
    if hps.critic_reg_type:
        reg_info += '+' + str(hps.critic_reg_cf) + ' ' + hps.critic_reg_type
    fig.suptitle(r'%s%s, $C(x)=\psi_0 x + \psi_1, \,Q(x)=\delta_\theta(x)$' % (hps.loss_type, reg_info))

    traj_plots = []
    margin_x, margin_y, margin_z = np.abs(trajectory).max(axis=(0, 1))
    for i in range(trajectory.shape[1]):
        traj_plots.append(ax.plot(*trajectory[:1, i, :].T, c=cmap(i), animated=True, alpha=0.8)[0])
        init = trajectory[0, i]
        ax.plot([init[0]], [init[1]], [init[2]], color=cmap(i), marker='o', ms=10)

    ax.set_xlim(-margin_x, margin_x)
    ax.set_ylim(-margin_y, margin_y)
    ax.set_zlim(-margin_z, margin_z)

    if hps.loss_type == 'W1':
        margin = margin_y * 1.5
        y = np.linspace(-margin, margin, 128)
        ax.plot([0] * 128, y, [hps.dirac_target] * 128, '--', color='orange', lw=3)
    elif hps.loss_type == 'JSD':
        margin = margin_x * 1.5
        margin1 = margin_y * 1.5
        x = np.linspace(-margin, margin, 128)
        y = - hps.dirac_target * x
        ax.plot(x, y, [hps.dirac_target,] * 128, '--', color='orange', lw=3, alpha=0.3)
        ax.plot([0] * 128, np.linspace(-margin1, margin1, 128), [hps.dirac_target,] * 128, '--', color='orange', lw=3, alpha=0.3)
        ax.plot([0], [0], [hps.dirac_target], color='orange', marker='*', ms=10, label='target')
    else:
        raise NotImplementedError(f'{hps.loss_type}')

    def draw(n):
        for i in range(trajectory.shape[1]):
            traj_plots[i].set_data_3d(*trajectory[:1+n * 10, i, :].T)
        return traj_plots

    anim = animation.FuncAnimation(fig, draw, frames=250, interval=20, blit=True)
    return anim
