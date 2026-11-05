"""
The ROSAME objective (Xi, Gould and Thiebaux, ICAPS 2024) for padded batches of traces.

For a transition s1 --a--> s2 with grounded (pre, add, del) of a, per proposition:

    prediction     (s1 * (1 - del) + (1 - s1) * add  -  s2)^2     the action predicts s2 from s1
    applicability  ((1 - s1) * pre)^2                              its preconditions hold in s1
    prior          prior_weight * (1 - pre)^2                      prefer preconditions (lambda)

States may be probabilities (e.g. outputs of a perception model) or 0/1 values. The prior resolves
facts that are true before and after every occurrence of an action, which the data alone cannot
classify as precondition or irrelevant; ROSAME's default treats them as preconditions. It also makes
learning robust to noise: a fact is kept as a precondition while it is false in fewer than roughly a
`prior_weight` fraction of the action's occurrences.

Hyperparameters (paper notation in brackets):
    prior_weight     [lambda] weight of the prior, default 0.2; 0 disables it
    endpoint_weight  [gamma]  extra weight of the terms that involve each trace's first state
                     (applicability of its first transition) and last state (prediction of its last
                     transition), for settings where those two states are known more reliably than
                     the intermediate ones; default 1.0 (no extra weight)
    reduction        how entries of different traces are combined:
                     "transition" (default) every real (transition, proposition) entry counts
                         equally: per-proposition terms are summed and divided by the number of
                         real transitions. Use this when traces differ in size.
                     "trace" each trace is averaged over its own T_b * P_b entries, then traces are
                         averaged: the paper's MSE, and equal to running the loss per trace and
                         averaging. With different object sets it weights a transition by
                         1 / (T_b * P_b), so evidence from large traces is outweighed by the prior
                         from small ones.
                     With equal P and T the two differ only by a constant factor.

Only real entries count: padded propositions and steps (batch.prop_mask, batch.step_mask) are
excluded. The prior is summed over the propositions the action touches; for the others pre is the
constant 0, so including them (as the paper's formula does) adds a constant with no gradient.
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from rosame.grounding import GroundedBatch

REDUCTIONS = ("transition", "trace")


def step_weights(step_mask: torch.Tensor, endpoint_weight: float, first: bool = True, last: bool = True) -> torch.Tensor:
    """[B, T]: 1 on real steps, 0 on masked ones, times endpoint_weight on each trace's first and/or
    last real step (squared on a single-step trace when both apply)."""
    w = step_mask.to(torch.get_default_dtype()).clone()
    if endpoint_weight == 1.0:
        return w
    T = step_mask.shape[1]
    idx = torch.arange(T, device=step_mask.device).expand_as(step_mask)
    rows = torch.nonzero(step_mask.any(1)).squeeze(1)
    if first:
        w[rows, torch.where(step_mask, idx, T).min(1).values[rows]] *= endpoint_weight
    if last:
        w[rows, torch.where(step_mask, idx, -1).max(1).values[rows]] *= endpoint_weight
    return w


def rosame_loss(pre: torch.Tensor, add: torch.Tensor, dele: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
                batch: GroundedBatch, *, prior_weight: float = 0.2, endpoint_weight: float = 1.0,
                reduction: str = "transition", num_transitions: Optional[int] = None,
                return_terms: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """The ROSAME objective. pre, add, dele: ROSAME(batch) outputs, [B, T, P]; s1, s2: states before
    and after each transition, [B, T, P], values in [0, 1]. See the module docstring for the
    hyperparameters. num_transitions (reduction="transition" only) overrides the normaliser, e.g. the
    total over all chunks when accumulating gradients chunk by chunk. return_terms=True returns
    {"loss", "prediction", "applicability", "prior"} (each reduced the same way)."""
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    if not (s1.shape == s2.shape == pre.shape == add.shape == dele.shape):
        raise ValueError(f"shape mismatch: pre {tuple(pre.shape)}, s1 {tuple(s1.shape)}, s2 {tuple(s2.shape)}")
    batch = batch.to(pre.device)
    real = batch.step_mask[:, :, None] & batch.prop_mask[:, None, :]          # [B, T, P]
    zero = pre.new_zeros(())
    w = lambda **kw: step_weights(batch.step_mask, endpoint_weight, **kw).to(pre)[..., None]

    prediction = (s1 * (1 - dele) + (1 - s1) * add - s2).pow(2) * w(first=False)
    applicability = ((1 - s1) * pre).pow(2) * w(last=False)
    terms = {"prediction": torch.where(real, prediction, zero),
             "applicability": torch.where(real, applicability, zero)}
    if prior_weight:
        prior = prior_weight * (1 - pre).pow(2)
        terms["prior"] = torch.where(real & batch.touched(), prior, zero)
    else:
        terms["prior"] = torch.zeros_like(pre)

    if reduction == "trace":
        count = real.sum((1, 2)).clamp_min(1)
        reduce = lambda x: (x.sum((1, 2)) / count).mean()
    else:
        n = num_transitions if num_transitions is not None else max(1, batch.num_transitions)
        reduce = lambda x: x.sum() / n
    out = {k: reduce(v) for k, v in terms.items()}
    out["loss"] = out["prediction"] + out["applicability"] + out["prior"]
    return out if return_terms else out["loss"]


class ROSAMELoss(nn.Module):
    """The ROSAME objective as a module holding its hyperparameters.

        loss_fn = ROSAMELoss()                                  # paper defaults, transition reduction
        loss_fn = ROSAMELoss(prior_weight=0.1, endpoint_weight=5.0, reduction="trace")
        loss = loss_fn(pre, add, dele, s1, s2, batch)
    """

    def __init__(self, prior_weight: float = 0.2, endpoint_weight: float = 1.0, reduction: str = "transition"):
        super().__init__()
        if prior_weight < 0:
            raise ValueError("prior_weight must be >= 0")
        if endpoint_weight <= 0:
            raise ValueError("endpoint_weight must be > 0")
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
        self.prior_weight, self.endpoint_weight, self.reduction = prior_weight, endpoint_weight, reduction

    def forward(self, pre, add, dele, s1, s2, batch: GroundedBatch, num_transitions: Optional[int] = None,
                return_terms: bool = False):
        return rosame_loss(pre, add, dele, s1, s2, batch, prior_weight=self.prior_weight,
                           endpoint_weight=self.endpoint_weight, reduction=self.reduction,
                           num_transitions=num_transitions, return_terms=return_terms)

    def extra_repr(self) -> str:
        return f"prior_weight={self.prior_weight}, endpoint_weight={self.endpoint_weight}, reduction={self.reduction!r}"
