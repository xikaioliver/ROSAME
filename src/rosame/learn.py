"""
High-level learning: give a domain and traces, get a trained ROSAME and its discrete action model.

    import rosame
    result = rosame.learn(domain, traces)
    print(result.to_pddl())

Traces (planning_structs.Trace) may use different object sets and have different lengths. States may be symbolic (sets of
true atoms) or probabilistic (probabilities per proposition, e.g. from a classifier or noisy
sensors), which is what ROSAME is designed for. Everything here is a thin layer over the public
building blocks (Grounder, ROSAME, ROSAMELoss); write your own loop with those to customise
training, e.g. to train a perception model jointly.
"""
from __future__ import annotations

import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from rosame.grounding import GroundedBatch, Grounder
from rosame.losses import ROSAMELoss
from rosame.model import ROSAME
from planning_structs.action_model import ActionModel
from planning_structs.domain import Domain
from planning_structs.trace import Trace
from rosame.utils import stack_padded


@dataclass
class LearnResult:
    model: ROSAME
    history: List[Dict[str, float]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def action_model(self) -> ActionModel:
        return self.model.action_model()

    def to_pddl(self, **kwargs) -> str:
        """See rosame.io.write_pddl."""
        return self.model.to_pddl(**kwargs)


def prepare_traces(grounder: Grounder, traces: Sequence[Trace], on_unknown: str = "warn"):
    """Encode states and ground actions. Returns (encoded, stats); each encoded item is
    (objects, states [T + 1, P], grounded actions with None for ignored steps)."""
    encoded, counts = [], Counter()
    unlabelled = repeated = 0
    for k, tr in enumerate(traces):
        objects = [tuple(o) for o in tr.objects]
        pos = {n: i for i, (n, _) in enumerate(objects)}
        if isinstance(tr.states, torch.Tensor):
            states = tr.states.float()
            P = grounder.propositions(objects).num_props
            if states.dim() != 2 or states.shape[1] != P:
                raise ValueError(f"trace {k}: states tensor has shape {tuple(states.shape)}, expected [T + 1, {P}]")
        else:
            states = torch.stack([grounder.encode_state(objects, s, pos, on_unknown) for s in tr.states])
        if states.shape[0] != len(tr.actions) + 1:
            raise ValueError(f"trace {k}: {states.shape[0]} states for {len(tr.actions)} actions (need T + 1)")
        if states.numel() and (states.min() < 0 or states.max() > 1):
            raise ValueError(f"trace {k}: state values must be probabilities in [0, 1]")
        calls = []
        for a in tr.actions:
            g = None if a is None else grounder.ground_action(objects, a, pos)
            unlabelled += a is None
            repeated += a is not None and g is None
            if g is not None:
                counts[grounder.schemas[g.schema].name] += 1
            calls.append(g)
        encoded.append((objects, states, calls))
    P = [s.shape[1] for _, s, _ in encoded]
    stats = {"traces": len(encoded), "transitions": sum(len(c) for _, _, c in encoded),
             "used_transitions": sum(counts.values()), "ignored_unknown_action": unlabelled,
             "ignored_repeated_objects": repeated,
             "transitions_per_schema": {a.name: counts.get(a.name, 0) for a in grounder.schemas},
             "propositions": [min(P, default=0), max(P, default=0)]}
    stats["schemas_without_data"] = [n for n, c in stats["transitions_per_schema"].items() if c == 0]
    return encoded, stats


def _chunks(encoded, grounder: Grounder, max_elements: int, device) -> List[Tuple[GroundedBatch, torch.Tensor, int]]:
    """Collate once: traces sorted by size, packed greedily under the element budget."""
    order = sorted(range(len(encoded)), key=lambda i: encoded[i][1].numel())
    groups, cur = [], []
    cost = lambda idx: len(idx) * max(encoded[i][1].shape[0] for i in idx) * max(encoded[i][1].shape[1] for i in idx)
    for i in order:
        if cur and cost(cur + [i]) > max_elements:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    out = []
    for idx in groups:
        gb = grounder.collate([encoded[i][0] for i in idx], [encoded[i][2] for i in idx])
        out.append((gb.to(device), stack_padded([encoded[i][1] for i in idx]).to(device), len(idx)))
    return out


def learn(domain: Domain, traces: Sequence[Trace], *, loss: Optional[ROSAMELoss] = None, epochs: int = 2000,
          min_epochs: int = 200, patience: int = 200, lr: float = 1e-3, seed: Optional[int] = 0,
          device: str = "cpu", max_elements: int = 4_000_000, model: Optional[ROSAME] = None,
          model_kwargs: Optional[dict] = None, on_unknown_atom: str = "warn", verbose: bool = False,
          log_every: int = 100) -> LearnResult:
    """Train ROSAME on traces with full-batch Adam.

    loss           a ROSAMELoss (defaults: prior_weight 0.2, endpoint_weight 1.0, reduction "transition")
    epochs         maximum number of optimisation steps; each step uses every trace
    min_epochs, patience
                   stop once the most likely role of every lifted slot has not changed for
                   `patience` steps, after at least `min_epochs` steps
    lr, seed       Adam learning rate; seed for the model initialisation (None: leave the RNG alone)
    device         e.g. "cpu" or "cuda"
    max_elements   memory budget per chunk in (states x propositions) entries; a step is split into
                   chunks with gradient accumulation, which leaves the full-batch gradient unchanged
    model          continue training an existing ROSAME instead of creating one
    model_kwargs   arguments for a new ROSAME (network sizes etc.)
    on_unknown_atom
                   atoms with no proposition (e.g. repeated objects): "warn", "ignore" or "raise"
    """
    t0 = time.perf_counter()
    loss_fn = loss if loss is not None else ROSAMELoss()
    grounder = Grounder(domain)
    encoded, stats = prepare_traces(grounder, traces, on_unknown_atom)
    if stats["used_transitions"] == 0:
        raise ValueError("no usable transitions")
    if stats["schemas_without_data"]:
        warnings.warn(f"no usable transitions for action schema(s) {stats['schemas_without_data']}; their "
                      f"learned roles only reflect the initialisation", stacklevel=2)
    if stats["ignored_repeated_objects"]:
        warnings.warn(f"{stats['ignored_repeated_objects']} transition(s) ignored because their action repeats an "
                      f"object (ROSAME binds distinct objects to distinct parameters)", stacklevel=2)

    if model is None:
        if seed is not None:
            torch.manual_seed(seed)
        model = ROSAME(domain, **(model_kwargs or {}))
    model = model.to(device)
    chunks = _chunks(encoded, grounder, max_elements, device)
    n_traces, n_real = len(encoded), stats["used_transitions"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history, last, stable, epoch = [], model.learned_roles(), 0, 0
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        totals = Counter()
        for gb, states, n in chunks:
            pre, add, dele = model(gb)
            s1, s2 = states[:, :-1], states[:, 1:]
            if loss_fn.reduction == "transition":
                terms = loss_fn(pre, add, dele, s1, s2, gb, num_transitions=n_real, return_terms=True)
            else:
                terms = {k: v * (n / n_traces) for k, v in loss_fn(pre, add, dele, s1, s2, gb, return_terms=True).items()}
            terms["loss"].backward()
            totals.update({k: v.item() for k, v in terms.items()})
        opt.step()
        roles = model.learned_roles()
        stable = stable + 1 if torch.equal(roles, last) else 0
        last = roles
        done = epoch >= min_epochs and stable >= patience
        if epoch == 1 or epoch % log_every == 0 or done or epoch == epochs:
            history.append({"epoch": epoch, **dict(totals)})
            if verbose:
                print(f"epoch {epoch:5d}  loss {totals['loss']:.6f}  (prediction {totals['prediction']:.6f}, "
                      f"applicability {totals['applicability']:.6f}, prior {totals['prior']:.6f})  "
                      f"roles unchanged for {stable}")
        if done:
            break
    stats.update(epochs=epoch, stopped_early=epoch < epochs, chunks=len(chunks),
                 seconds=round(time.perf_counter() - t0, 2))
    return LearnResult(model.eval(), history, stats)
