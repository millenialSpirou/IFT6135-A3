r"""
:mod:`utils.train` -- Miscellaneous boilerplate code for training/eval time
================================================================================

.. module:: train
   :platform: Unix

"""
from contextlib import contextmanager
from collections import defaultdict
import random

import numpy
import torch


class average_meter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, values):
        n = values.shape[0]
        self.val = values.mean()
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class running_average_meter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


@contextmanager
def eval_ctx(*model, debug=False, no_grad=True):
    for m in model:
        m.eval()
    torch.autograd.set_detect_anomaly(debug)
    with torch.set_grad_enabled(mode=not no_grad):
        yield
    torch.autograd.set_detect_anomaly(False)
    for m in model:
        m.train()


def seed_prng(seed, use_cuda=False, deterministic=False):
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if use_cuda:
            torch.backends.cudnn.benchmark = True
    random.seed(seed)
    numpy.random.seed(random.randint(1, 100000))
    torch.random.manual_seed(random.randint(1, 100000))
    if use_cuda is True:
        torch.cuda.manual_seed_all(random.randint(1, 100000))


class accumulator(object):

    def __init__(self, init=None, mode='list'):
        init = init or dict()
        self.state = defaultdict(list)
        self.state.update(init)
        for k, v in self.state.items():
            if k == 'steps':
                dtype = int
            else:
                dtype = float
            if mode == 'list':
                try:
                    self.state[k] = v.tolist()
                except AttributeError:
                    self.state[k] = v
            elif mode == 'numpy':
                self.state[k] = np.asarray(v, dtype=dtype)
            elif mode == 'torch':
                self.state[k] = torch.Tensor(v, dtype=dtype)

    def __getattr__(self, k):
        return self.state[k]

    @property
    def to_torch(self):
        return accumulator(self.state, mode='torch')

    @property
    def to_numpy(self):
        return accumulator(self.state, mode='numpy')

    @property
    def to_list(self):
        return accumulator(self.state, mode='list')
