# -*- coding: utf-8 -*-
r"""
:mod:`optim` -- Optimization utility for training with Pytorch
==============================================================

.. module:: optim
   :platform: Unix
   :synopsis: This one just contains a builder for a torch.optim.Optimizer

"""
from functools import partial
from typing import (Tuple, Text)

from torch import optim


Vector = Tuple[float]


def init_optimizer(variables, type: Text,
                   lr: float=1e-3,
                   betas: Vector=(0.9, 0.999),
                   wd: float=1e-6,
                   eta: float=1e-3, eps: float=1e-8) -> optim.Optimizer:
    Algo = getattr(optim, type)
    momentum = betas[0]
    partial_algo = partial(Algo, variables, lr=lr, weight_decay=wd)
    if 'Adam' in type:
        return partial_algo(betas=betas, eps=eps)
    elif type == 'SGD':
        return partial_algo(momentum=momentum)
    else:
        raise NotImplementedError
