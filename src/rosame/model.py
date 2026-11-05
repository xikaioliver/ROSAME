"""
ROSAME: a lifted, probabilistic action model as a PyTorch module.

For every action schema, a small network (PAM) outputs a categorical distribution over four roles
for each of the schema's (predicate, binding) slots:

    ROLES = ("none", "add", "pre", "pre+del")

from which the probabilities of being a precondition, an add effect and a delete effect follow:

    pre = p(pre) + p(pre+del)      add = p(add)      del = p(pre+del)

so a slot is never both added and deleted, and a deleted fact is always a precondition (STRIPS).
Grounding (rosame.grounding) turns these lifted values into vectors over each trace's propositions;
ROSAME does this for a whole batch of traces with different object sets at once.

ROSAME contains no loss: it returns grounded (pre, add, del) vectors, so any objective can be used.
The objective of the ROSAME paper is rosame.losses.ROSAMELoss.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rosame.grounding import GroundedBatch, Grounder, lifted_slots, schema_triple_table
from rosame.planning_structs.action_model import ActionModel
from rosame.planning_structs.domain import ActionSchema, Domain, Predicate
from rosame.planning_structs.instance import Instance

ROLES = ("none", "add", "pre", "pre+del")
Triple = Tuple[ActionSchema, Predicate, Tuple[int, ...]]


class PAMNetwork(nn.Module):
    """Probabilistic action model network of one action schema: maps one input vector per slot to
    four role logits. Inputs are fixed random vectors by default (trainable_inputs=True learns them)."""

    def __init__(self, num_triples: int, dim: int = 128, hidden: int = 128, depth: int = 2,
                 dropout: float = 0.0, trainable_inputs: bool = False, use_layernorm: bool = True):
        super().__init__()
        self.num_triples = num_triples
        w = torch.randn(num_triples, dim)
        if trainable_inputs:
            self.inputs = nn.Parameter(w)
        else:
            self.register_buffer("inputs", w)
        layers: List[nn.Module] = []
        in_d = dim
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(in_d, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_d = hidden
        layers += [nn.Linear(in_d, 4)]
        self.mlp = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(dim) if use_layernorm else nn.Identity()

    def forward(self) -> torch.Tensor:
        """Role logits [num_triples, 4]."""
        return self.mlp(self.ln(self.inputs))


class DomainWrapper(nn.Module):
    """One PAMNetwork per action schema, and the fixed order of all lifted slots ("triples")."""

    def __init__(self, domain: Domain, pam_dim: int = 128, pam_hidden: int = 128, pam_depth: int = 3,
                 pam_dropout: float = 0.0, trainable_inputs: bool = False, use_layernorm: bool = True):
        super().__init__()
        self.domain = domain
        self.triples: List[Triple] = [(a, p, x) for a, slots in lifted_slots(domain).items() for p, x in slots]
        self._schema_to_triple_idxs: Dict[ActionSchema, List[int]] = {}
        for tidx, (a, _, _) in enumerate(self.triples):
            self._schema_to_triple_idxs.setdefault(a, []).append(tidx)
        self._schema_to_pam = nn.ModuleDict({
            a.name: PAMNetwork(len(self._schema_to_triple_idxs.get(a, [])), pam_dim, pam_hidden, pam_depth,
                               pam_dropout, trainable_inputs, use_layernorm)
            for a in domain.action_schemas})

    def pam_for(self, a: ActionSchema) -> PAMNetwork:
        return self._schema_to_pam[a.name]

    def num_triples_for(self, a: ActionSchema) -> int:
        return len(self._schema_to_triple_idxs.get(a, []))

    def logits(self) -> torch.Tensor:
        """Role logits of every lifted slot, [N, 4], in triple order."""
        chunks = [self.pam_for(a)() for a in self.domain.action_schemas if self.num_triples_for(a) > 0]
        if not chunks:
            return torch.empty(0, 4, device=next(self.parameters()).device)
        return torch.cat(chunks, dim=0)

    def forward_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pre, add, del) of every lifted slot, each [N]."""
        probs = F.softmax(self.logits(), dim=-1)
        return probs[:, 2] + probs[:, 3], probs[:, 1], probs[:, 3]


class ROSAME(nn.Module):
    """
    A lifted probabilistic action model for a Domain.

        model = ROSAME(domain)
        batch = Grounder(domain).collate(objects_per_trace, actions_per_trace)
        pre, add, dele = model(batch)          # each [B, T, P_max]; padded entries are exactly 0

    Every entry of the outputs is defined, including propositions the action does not touch (their
    pre/add/del are the constant 0), so a loss over all propositions also trains anything that
    produced the states (e.g. a perception network).

    Single-object-set use is still supported: model.set_instance(instance); model(action_indices)
    returns [len(action_indices), P] for that instance.

    Args:
        domain: the typed STRIPS domain (types, predicates, action schemas).
        pam_dim, pam_hidden, pam_depth, pam_dropout, trainable_inputs, use_layernorm: size and options
            of the per-schema networks (see PAMNetwork).
    """

    def __init__(self, domain: Domain, pam_dim: int = 128, pam_hidden: int = 128, pam_depth: int = 3,
                 pam_dropout: float = 0.0, trainable_inputs: bool = False, use_layernorm: bool = True):
        super().__init__()
        self.dw = DomainWrapper(domain, pam_dim, pam_hidden, pam_depth, pam_dropout, trainable_inputs, use_layernorm)
        self.triples: List[Triple] = self.dw.triples
        self.register_buffer("_schema_triples", schema_triple_table(domain), persistent=False)
        self._grounder = Grounder(domain)
        self._instance: Optional[Instance] = None

    @property
    def domain(self) -> Domain:
        return self.dw.domain

    # ----- lifted values -----

    def lifted_effects(self) -> torch.Tensor:
        """[3, N + 1]: rows pre, add, del of every lifted slot; column 0 is a constant 0 entry used
        for propositions an action does not touch."""
        pre, add, dele = self.dw.forward_all()
        return F.pad(torch.stack([pre, add, dele]), (1, 0))

    def role_probabilities(self) -> torch.Tensor:
        """[N, 4] probability of each role in ROLES, per lifted slot (triple order)."""
        return F.softmax(self.dw.logits(), dim=-1)

    @torch.no_grad()
    def learned_roles(self) -> torch.Tensor:
        """[N] index into ROLES of the most likely role of each lifted slot."""
        return self.role_probabilities().argmax(-1)

    @torch.no_grad()
    def action_model(self) -> ActionModel:
        """The discrete STRIPS model given by the most likely role of every slot."""
        model = ActionModel(self.domain)
        for (a, p, x), r in zip(self.triples, self.learned_roles().tolist()):
            if ROLES[r] in ("pre", "pre+del"):
                model.preconditions.setdefault(a.name, []).append((p, x))
            if ROLES[r] == "add":
                model.add_effects.setdefault(a.name, []).append((p, x))
            if ROLES[r] == "pre+del":
                model.del_effects.setdefault(a.name, []).append((p, x))
        return model

    def to_pddl(self, **kwargs) -> str:
        """The learned model as a PDDL domain (see rosame.io.write_pddl for options)."""
        from rosame.io.pddl import write_pddl
        return write_pddl(self.action_model(), **kwargs)

    # ----- grounding -----

    def forward(self, x, lifted: Optional[torch.Tensor] = None):
        """Ground a GroundedBatch -> (pre, add, del), each [*batch.schema.shape, P_max]. With an
        instance set (set_instance), x may instead be action indices into instance.actions.
        `lifted` (from lifted_effects()) reuses one network evaluation for several batches."""
        if isinstance(x, GroundedBatch):
            x = x.to(self._schema_triples.device)
            return self._ground(x.schema, x.props, x.num_props_max, lifted)
        if self._instance is None:
            raise TypeError("pass a GroundedBatch, or call set_instance(instance) before passing action indices")
        idx = torch.as_tensor([x] if isinstance(x, int) else x, dtype=torch.long).reshape(-1)
        schema, props, _ = self._grounder._dense_table(self._instance)
        dev = self._schema_triples.device
        return self._ground(schema[idx].to(dev), props[idx].to(dev), self._instance.num_propositions, lifted)

    def _ground(self, schema: torch.Tensor, props: torch.Tensor, num_props: int, lifted: Optional[torch.Tensor]):
        L = self.lifted_effects() if lifted is None else lifted
        vals = L[:, self._schema_triples[schema]]                       # [3, *, K]
        idx = props.unsqueeze(0).expand_as(vals)
        out = vals.new_zeros((3, *schema.shape, num_props)).scatter_add(-1, idx, vals)
        return out[0], out[1], out[2]

    # ----- single-instance convenience -----

    def set_instance(self, instance: Instance) -> None:
        """Use model(action_indices) with indices into instance.actions."""
        self._instance = instance

    def get_current_instance(self) -> Optional[Instance]:
        return self._instance
