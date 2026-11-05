"""Helpers for padded batches: stacking ragged tensors, assembling state sequences, and a sampler
that groups traces of similar size to limit padding."""
from __future__ import annotations

import math
import random
from typing import Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


def stack_padded(tensors: Sequence[torch.Tensor], value: float = 0.0) -> torch.Tensor:
    """Pad every dimension to the largest size in the list (at the end), then stack on a new dim 0.
    E.g. states [P_b] -> [B, P_max]; [T_b, P_b] -> [B, T_max, P_max]; images [T_b, C, H, W] -> [B, T_max, C, H, W]."""
    shape = [max(t.shape[d] for t in tensors) for d in range(tensors[0].dim())]
    out = tensors[0].new_full((len(tensors), *shape), value)
    for b, t in enumerate(tensors):
        out[(b, *(slice(0, s) for s in t.shape))] = t
    return out


def assemble_states(init: torch.Tensor, inter: torch.Tensor, final: torch.Tensor,
                    num_steps: torch.Tensor) -> torch.Tensor:
    """Build the state sequence of each trace from its initial state [B, P], predicted intermediate
    states [B, T_max - 1, P] and final state [B, P], with the final state placed after the trace's
    own last step: returns [B, T_max + 1, P] with states[b, num_steps[b]] = final[b]. Then
    s1 = states[:, :-1] and s2 = states[:, 1:]; entries past num_steps[b] are padding."""
    B, P = init.shape
    states = torch.cat([init[:, None], inter, init.new_zeros(B, 1, P)], dim=1)
    idx = num_steps.to(torch.long).view(B, 1, 1).expand(B, 1, P)
    return states.scatter(1, idx, final[:, None])


class BucketBatchSampler(Sampler[List[int]]):
    """Batches of traces with similar proposition counts, to limit padding when object-set sizes
    vary a lot (P grows quadratically with objects for binary predicates). Indices are shuffled,
    split into pools of `pool` batches, sorted by size inside each pool, cut into batches, and the
    batches are shuffled. Use as DataLoader(dataset, batch_sampler=BucketBatchSampler(...))."""

    def __init__(self, sizes: Sequence[int], batch_size: int, pool: int = 50, shuffle: bool = True,
                 drop_last: bool = False, seed: int = 0):
        self.sizes, self.batch_size, self.pool = list(sizes), batch_size, pool
        self.shuffle, self.drop_last, self.seed, self.epoch = shuffle, drop_last, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        idx = list(range(len(self.sizes)))
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        span = self.batch_size * self.pool
        for i in range(0, len(idx), span):
            chunk = sorted(idx[i:i + span], key=lambda j: self.sizes[j])
            batches += [chunk[k:k + self.batch_size] for k in range(0, len(chunk), self.batch_size)]
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        n = len(self.sizes)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
