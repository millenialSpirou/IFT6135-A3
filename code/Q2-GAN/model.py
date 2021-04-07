# -*- coding: utf-8 -*-
r"""
:mod:`model` -- Implements neural network architecture for image generation
===========================================================================

.. module:: model
   :platform: Unix
   :synopsis: Implements a NN generator and critic

"""
from functools import partial

import torch
from torch import nn


def sn_conv2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))


class ResBlock(nn.Module):
    def __init__(self, dim, style='anhbnh', learnable_alpha=None,
                 sn=False, activation='ReLU'):
        super().__init__()

        Conv2d = sn_conv2d if sn is True else nn.Conv2d

        def build_part(s):
            if s == 'n':
                p = nn.BatchNorm2d(dim)
            elif s == 'h':
                p = nn.ReLU(inplace=True) if activation == 'ReLU' else nn.LeakyReLU(0.2, inplace=True)
            elif s == 'a':
                p = Conv2d(dim, dim, 3, 1, 1)
            elif s == 'b':
                p = Conv2d(dim, dim, 1)
            else:
                raise ValueError
            return p

        self.block = nn.Sequential(*[build_part(s) for s in style])

        self.alpha = 0.
        if learnable_alpha is not None:
            self.alpha = nn.Parameter(torch.tensor([float(learnable_alpha),]))

    def forward(self, x):
        cf = torch.sigmoid(self.alpha)
        return (1 - cf) * x + cf * self.block(x)


class Critic(nn.Module):
    def __init__(self, dimh=64, sn=False):
        super(Critic, self).__init__()

        Conv2d = sn_conv2d if sn is True else nn.Conv2d
        build_res = partial(ResBlock, style='ahbh',
                            learnable_alpha=-3., sn=sn,
                            activation='LeakyReLU')

        encoder = [Conv2d(3, dimh, 4, 2, 1),  # 32 -> 16
                   nn.LeakyReLU(0.2, inplace=True),
                   build_res(dimh),
                   Conv2d(dimh, 2*dimh, 4, 2, 1),  # 16 -> 8
                   nn.LeakyReLU(0.2, inplace=True),
                   build_res(2*dimh),
                   Conv2d(2*dimh, 4*dimh, 4, 2, 1),  # 8 -> 4
                   nn.LeakyReLU(0.2, inplace=True),
                   build_res(4*dimh),
                   Conv2d(4*dimh, 1, 4, 1, 0)]

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        return self.encoder(x).squeeze()


class Generator(nn.Module):
    def __init__(self, dimz=100, dimh=64, default_batch_size=1):
        super(Generator, self).__init__()
        self.dimz = dimz
        self.batch_size = default_batch_size

        build_res = partial(ResBlock, style='anhbnh',
                            learnable_alpha=-3.,
                            activation='ReLU')

        decoder = [nn.ConvTranspose2d(dimz, 4*dimh, 4, 1, 0),
                   nn.BatchNorm2d(4*dimh),
                   nn.ReLU(True),
                   build_res(4*dimh),
                   nn.ConvTranspose2d(4*dimh, 2*dimh, 4, 2, 1),
                   nn.BatchNorm2d(2*dimh),
                   nn.ReLU(True),
                   build_res(2*dimh),
                   nn.ConvTranspose2d(2*dimh, dimh, 4, 2, 1),
                   nn.BatchNorm2d(dimh),
                   nn.ReLU(True),
                   build_res(dimh),
                   nn.ConvTranspose2d(dimh, 3, 4, 2, 1),
                   nn.Tanh()
                   ]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, z=None):
        if z is None:
            z = torch.randn(self.batch_size, self.dimz,
                            device=self.decoder[0].weight.device)
        return self.decoder(z.view(-1, self.dimz, 1, 1))
