# pseudo_label.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import torch


@dataclass
class PseudoLabels:
    """
    Stores pseudo-labels produced by the constraint optimization solver.

    - traces[idx] = (flag, state_label, action_label)
        where `flag` is kept for backward compatibility (legacy code used 1).
    - model is a dict: schema_name -> tensor (#preds, 4) one-hot label
    """
    traces: Dict[int, Tuple[int, torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    model: Optional[Dict[str, torch.Tensor]] = None

    def clear(self) -> None:
        self.traces.clear()
        self.model = None

    def update(
        self,
        indices: List[int],
        state_labels: List[torch.Tensor],
        action_labels: List[torch.Tensor],
        model_label: Dict[str, torch.Tensor],
    ) -> None:
        """
        Update trace-level pseudo-labels and overwrite the stored model label.

        This mirrors legacy semantics:
          - each idx gets (1, state_label, action_label)
          - model_label overwrites stored model for each schema
        """
        for i, idx in enumerate(indices):
            self.traces[int(idx)] = (1, state_labels[i], action_labels[i])

        if self.model is None:
            self.model = model_label
        else:
            for schema in self.model:
                self.model[schema] = model_label[schema]

    # ---- convenience (optional) ----

    def has(self, idx: int) -> bool:
        return int(idx) in self.traces

    def get(self, idx: int) -> Optional[Tuple[int, torch.Tensor, torch.Tensor]]:
        return self.traces.get(int(idx), None)

    def __len__(self) -> int:
        return len(self.traces)
