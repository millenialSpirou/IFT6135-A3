r"""
:mod:`utils.data` -- Miscellaneous boilerplate code for data prepare and load
================================================================================

.. module:: data
   :platform: Unix
   :synopsis: Infinite sampler and dataset factory
   :author: Christos Tsirigotis

"""
from abc import (ABCMeta, abstractmethod, abstractproperty)
import os

import numpy
import torch

from utils import (State, seed_prng)
from utils.meta import Factory


class MultiEpochSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly over multiple epochs
    Arguments:
        data_source (Dataset): dataset to sample from
        num_epochs (int): Number of times to loop over the dataset
        start_itr (int): which iteration to begin from
        batch_size (int): how many indices sampler should return

    """

    def __init__(self, data_source,
                 batches_seen=0, batch_size=128):
        self.n_data = len(data_source)
        self.batches_seen = batches_seen
        self.batch_size = batch_size

    def __iter__(self):
        def generate_indices(n, b_seen, b_size):
            s_seen = b_seen * b_size
            e_seen = s_seen // n
            extra = s_seen % n
            for _ in range(e_seen):
                numpy.random.permutation(n)
            rand_indices = numpy.random.permutation(n)
            rand_indices = rand_indices[extra:]
            while True:
                for index in rand_indices:
                    yield index
                rand_indices = numpy.random.permutation(n)

        return iter(generate_indices(self.n_data,
                                     self.batches_seen, self.batch_size))

    def __len__(self):
        raise NotImplementedError()


def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed
    # This value is determined by main process RNG and the worker id.
    seed_prng(seed + worker_id, use_cuda=State().use_cuda,
              deterministic=State().deterministic)


def prepare_splits(splits, N):
    splits = numpy.asarray(splits, dtype=numpy.float64)
    splits = numpy.around((splits / sum(splits)) * N).astype(int)
    s = sum(splits) - N
    if s > 0:
        i = numpy.argmax(splits)
    else:
        i = numpy.argmin(splits)
    splits[i] = splits[i] - s
    return tuple(splits.tolist())


class AbstractDataset(object, metaclass=ABCMeta):

    def __init__(self, root, num_threads=1, download=False, load=True,
                 splits=(1,), batch_size=1, mode='train', shuffle=True,
                 preload_to_gpu=False, **options):
        try:
            self.state = State()
            self.is_cuda = self.state.use_cuda
        except TypeError:
            self.state = None
            self.is_cuda = False

        self.root = os.path.abspath(os.path.expanduser(root))
        assert(num_threads >= 0)
        self.num_threads = num_threads
        self.splits = splits
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        self.preload_to_gpu = preload_to_gpu
        self.options = options
        self.options.update(batch_size=batch_size, mode=mode, shuffle=shuffle,
                            preload_to_gpu=preload_to_gpu)

        if download is True and self.check_exists(self.root) is not True:
            self.download(self.root)

        self._data = []
        if load is True:
            self.load()

    def download(self, root):
        pass

    def check_exists(self, root):
        return True

    @abstractmethod
    def prepare(self, root, **options):
        """Return a `torch.utils.data.Dataset` implementation."""
        pass

    def transform(self, batch):
        return batch

    @property
    def data(self):
        if not self._data or \
                any(not isinstance(ds, torch.utils.data.Dataset) for ds in self._data):
            raise ValueError("Call `load` method first.")
        return self._data

    @property
    def N(self):
        return self.n_data

    @abstractproperty
    def shape(self):
        pass

    def load(self):
        if self.check_exists(self.root) is False:
            raise RuntimeError(self.__class__.__name__ + ' not found.' +
                               ' You can use download=True to download it')
        self._data = self.prepare(self.root, **self.options)
        self.n_data = len(self._data)
        self.splits = prepare_splits(self.splits, self.n_data)
        self.n_splits = len(self.splits)
        self._data = torch.utils.data.random_split(self._data, self.splits)

    def build_loader(self, batch_size, sampler, split=0):
        """Return a `torch.utils.data.DataLoader` interface."""
        return torch.utils.data.DataLoader(dataset=self.data[split],
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           drop_last=False,
                                           num_workers=(self.num_threads if not self.preload_to_gpu else 0),
                                           pin_memory=(self.is_cuda and not self.preload_to_gpu),
                                           worker_init_fn=_worker_init_fn,
                                           )

    def _fetch(self, loader, stream):
        if self.is_cuda:
            with torch.cuda.stream(stream):
                batch = next(loader)
                batch = [x.cuda(non_blocking=True) for x in batch]
                batch = self.transform(batch)
        else:
            batch = next(loader)
            batch = self.transform(batch)
        return batch

    def sampler(self, split=0, project=None, infinite=True):
        fetch_stream = None
        if self.is_cuda:
            fetch_stream = torch.cuda.Stream()
        batches_seen = 0
        if infinite:
            sampler = MultiEpochSampler(self.data[split],
                                        batches_seen,
                                        self.batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(self.data[split])
        loader = iter(self.build_loader(self.batch_size, sampler, split=split))
        next_batch = self._fetch(loader, fetch_stream)
        try:
            while True:
                if self.is_cuda:
                    torch.cuda.current_stream().wait_stream(fetch_stream)
                current_batch = next_batch
                if self.is_cuda:
                    for x in current_batch:
                        x.record_stream(torch.cuda.current_stream())
                next_batch = self._fetch(loader, fetch_stream)
                yield current_batch if project is None else current_batch[project]
        except StopIteration:
            return current_batch if project is None else current_batch[project]


class Dataset(AbstractDataset, metaclass=Factory): pass
