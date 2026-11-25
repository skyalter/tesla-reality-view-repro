"""Which samples a training step sees, as a pure function of (seed, step, rank, number of ranks).

Every epoch is one fixed permutation of the dataset, derived from the seed and the epoch number. The
permutation is dealt out to the ranks like cards, so ranks never share a sample within an epoch. A step is a
position in the endless sequence of epochs; nothing else is remembered, which makes resuming exact.
"""
from __future__ import annotations

import hashlib

import numpy as np


def epoch_permutation(length: int, seed: int, epoch: int) -> np.ndarray:
    digest = hashlib.sha256(f"realityview-sampler:{seed}:{epoch}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], "little"))).permutation(length)


class StepBatchSampler:
    """Yields one list of dataset indices per step, forever, starting at `start_step`."""

    def __init__(self, length: int, batch_size: int, rank: int, world_size: int, seed: int, start_step: int = 0):
        if length < world_size:
            raise ValueError(f"dataset has {length} samples but there are {world_size} processes")
        self.length, self.batch_size = length, batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.per_rank = length // world_size          # samples one rank sees per epoch; the remainder is dropped
        self.start_step = start_step
        self._cache: tuple[int, np.ndarray] | None = None

    def _rank_epoch(self, epoch: int) -> np.ndarray:
        if self._cache is None or self._cache[0] != epoch:
            perm = epoch_permutation(self.length, self.seed, epoch)
            self._cache = (epoch, perm[self.rank::self.world_size][: self.per_rank])
        return self._cache[1]

    def batch_at(self, step: int) -> list[int]:
        first = step * self.batch_size
        return [int(self._rank_epoch(p // self.per_rank)[p % self.per_rank]) for p in range(first, first + self.batch_size)]

    def __iter__(self):
        step = self.start_step
        while True:
            yield self.batch_at(step)
            step += 1
