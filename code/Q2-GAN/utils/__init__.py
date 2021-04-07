r"""
:mod:`utils` -- Miscellaneous boilerplate code for training with Pytorch
================================================================================

.. module:: utils
   :platform: Unix
   :synopsis: Configuration of logger stream and a global experiment state
   :author: Christos Tsirigotis

"""
import os
import logging

import torch


from utils.train import seed_prng
from utils.meta import SingletonType


def config_logger(logpath, displaying=True, saving=True, debug=False):
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logger = logging.getLogger('Exp')
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(name)s:: time %(asctime)s | %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


class State(object, metaclass=SingletonType):

    def __init__(self, seed, args):
        self.logger = logging.getLogger('Exp')
        self.args = args
        self.seed = seed
        self.gpu = -1  # CPU is used
        if args.cuda is not None and torch.cuda.is_available():
            self.gpu = int(args.cuda)
        elif args.cuda is not None and not torch.cuda.is_available():
            self.logger.warning('CUDA is not available, but cuda args is not None: ' + str(args.cuda))
        self.device = self.gpu
        self.dtype = getattr(self.args, 'fp', '32')
        self.cvt = lambda x: x.to(device=self.device, dtype=self.dtype,
                                  memory_format=torch.contiguous_format)
        self.deterministic = args.deterministic
        seed_prng(self.seed, use_cuda=self.use_cuda,
                  deterministic=self.deterministic)

    @property
    def use_cuda(self):
        return self.gpu >= 0

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device_: int):
        if device_ >= 0:
            self._device = torch.device('cuda', device_)
            self.logger.info("Using CUDA device: %d", device_)
        else:
            self._device = torch.device('cpu')
            self.logger.info("Using CPU")

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, prec):
        if prec == '32':
            self._dtype = torch.float32
        elif prec == '64':
            self._dtype = torch.float64
        else:
            raise NotImplementedError(prec)
        torch.set_default_dtype(self._dtype)
        self.logger.info("Default floating precision: %s", self._dtype)

    def convert(self, x):
        return x.to(device=self.device, dtype=self.dtype,
                    memory_format=torch.contiguous_format)
