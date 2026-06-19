# selector.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence


@dataclass
class TraceSelector:
    """
    Simple FIFO selector for traces during training.

    Rationale:
      - We receive traces in (effectively) random order already due to shuffled dataloader.
      - We therefore select the first `capacity` traces encountered in the current cycle.
      - We then:
          * collect_traces()  -> build CP objectives from these traces
          * collect_indices() -> map CP solution back to the correct dataset indices
          * clear()           -> start a new cycle

    This consolidates the old best/worst heaps into one buffer.
    """
    capacity: int
    _indices: List[int] = field(default_factory=list, init=False, repr=False)
    _traces: List[Any] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {self.capacity}")

    def clear(self) -> None:
        """Reset the selector for a new collection cycle."""
        self._indices.clear()
        self._traces.clear()

    def full(self) -> bool:
        """Whether we have collected `capacity` traces."""
        return len(self._indices) >= self.capacity

    def update(self, indices, state_preds, action_preds, inits, goals) -> None:
        """
        Legacy-compatible signature. `losses` is ignored.

        Arguments are typically minibatch tensors/lists:
          - indices: shape [B] (tensor or list)
          - state_obs_traces: shape [B, ...] (tensor/list of tensors)
          - action_obs_traces: shape [B, ...]
          - inits: shape [B, ...]
          - goals: shape [B, ...]
        """
        if self.full():
            return

        # Determine how many from this batch we can still take
        remaining = self.capacity - len(self._indices)

        # `indices` can be a torch tensor; len(indices) works in both cases
        take = min(remaining, len(indices))
        for b in range(take):
            # store python int index
            idx_b = indices[b].item()
            # detach tensors to break autograd graph
            s_b = state_preds[b].detach()
            a_b = action_preds[b].detach()
            i_b = inits[b].detach()
            g_b = goals[b].detach()

            self._indices.append(idx_b)
            self._traces.append((s_b, a_b, i_b, g_b))

    def collect_indices(self) -> Sequence[int]:
        """
        Return the dataset indices corresponding to the collected traces.
        Use this to map CP solutions back to pseudo-label slots.
        """
        return tuple(self._indices)

    def collect_traces(self) -> Sequence[Any]:
        """
        Return the collected trace objects (in the same order as indices).
        Use this to build the CP problem objectives.
        """
        return tuple(self._traces)

    def __len__(self) -> int:
        return len(self._indices)
