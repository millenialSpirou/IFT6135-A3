# -*- coding: utf-8 -*-
r"""
:mod:`data` -- Defines possible dataset image tasks
===================================================

.. module:: data
   :platform: Unix
   :synopsis: Wrap AbstractDataset for torchvision datasets, like CIFAR10

"""
import os

import torch
from torch.utils.data import (TensorDataset, DataLoader)
import torchvision
from torchvision import transforms as tform

from utils.data import (AbstractDataset, Dataset)


class _Torchvision(AbstractDataset):

    def _get_data_class(self):
        dname = self.__class__.__name__
        dclass = getattr(torchvision.datasets, dname)
        assert(dclass is not None)
        return dclass

    def download(self, root):
        dclass = self._get_data_class()
        root = os.path.join(root, self.__class__.__name__)
        dclass(root, download=True)

    def check_exists(self, root):
        return None  # Delegate to torchvision object

    def prepare(self, root, mode='train', preload_to_gpu=False,
                transform=None, target_transform=None, **options):
        train = True if mode == 'train' else False
        dclass = self._get_data_class()
        root = os.path.join(root, self.__class__.__name__)

        if transform is None:
            transform = tform.ToTensor()
        data = dclass(root, train=train, transform=transform,
                      target_transform=target_transform)

        if preload_to_gpu:
            assert(self.state.use_cuda)
            N = len(data)
            size = (N, ) + self.shape
            preloaded_x = torch.zeros(*size, dtype=torch.float32, device=self.state.device)
            preloaded_y = torch.zeros(N, dtype=torch.int32, device=self.state.device)
            for i, (x, y) in enumerate(DataLoader(data, batch_size=1024)):
                preloaded_x[i * 1024: (i + 1) * 1024] = x.to(device=self.state.device)
                preloaded_y[i * 1024: (i + 1) * 1024] = y.to(device=self.state.device)
            preloaded_x.to(self.state.dtype)
            return TensorDataset(preloaded_x, preloaded_y)

        return data

    def transform(self, batch):
        # Scale [0, +1] to [-1, +1]
        sample = batch[0]
        dtype = sample.dtype
        mean = torch.as_tensor((0.5, 0.5, 0.5), dtype=dtype, device=sample.device)
        std = torch.as_tensor((0.5, 0.5, 0.5), dtype=dtype, device=sample.device)
        sample.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return batch


class CIFAR10(_Torchvision):

    @property
    def shape(self):
        return (3, 32, 32)
