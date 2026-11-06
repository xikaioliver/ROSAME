"""
Grounding lifted action-model slots onto object sets, for batched training.

ROSAME's parameters are lifted: for every action schema, a probability over four roles for each of
its (predicate, binding) slots. Grounding an action a(o1, ..., ok) in an object set writes those slot
values into a vector over that object set's propositions, at the propositions p(o_x1, ...) of its
slots; every other entry is 0. The values depend only on the schema; the positions depend only on
the object set. A batch of transitions over *different* object sets is therefore described by two
integer tensors, built here once on the CPU (e.g. in a DataLoader's collate_fn):

    schema [B, T]     the schema of each transition
    props  [B, T, K]  for each of its slots, the proposition index in its own trace's ordering

and ROSAME grounds the whole batch with one gather and one scatter (rosame.model). Propositions of
each trace are padded to the largest count in the batch, and traces to the longest; `prop_mask` and
`step_mask` mark what is real.

Only the actions that occur are grounded, so object sets with very many possible groundings are
fine. Proposition order follows planning_structs.Instance and depends only on the sequence of
object types (the object-set *signature*), so tables are cached per signature.

Atoms, actions and states can be given in several forms:
    atom    ("in", ("s1", "r2")) | ("in", "s1", "r2") | "(in s1 r2)" | "in s1 r2" | ("powered",)
    atom    ... or a planning_structs.Proposition
    action  the same forms with an action schema name, a planning_structs.Action, an integer (an
            index into Instance(domain, objects).actions, or into a declared action order), or None
            for a step to ignore
Data in its own order is handled by Grounder.reorderer (proposition order of vectors) and
Grounder.action_index_map / collate(..., action_orders=...) (integer actions).
    state   an iterable of true atoms | a dict {atom: probability} | a tensor [P] of probabilities
"""
from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass, fields
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from planning_structs.domain import ActionSchema, Domain, Predicate
from planning_structs.instance import Action, Instance, Proposition

Signature = Tuple[str, ...]
ObjectsLike = Union[Instance, Sequence[Tuple[str, str]]]
Atom = Tuple[str, Tuple[str, ...]]


# ---------------------------------------------------------------------------
# Lifted slot layout, shared with rosame.model
# ---------------------------------------------------------------------------

def lifted_slots(domain: Domain) -> Dict[ActionSchema, List[Tuple[Predicate, Tuple[int, ...]]]]:
    """Per action schema, its (predicate, binding) slots, in the order of ROSAME.triples."""
    return {a: [(p, x) for p in domain.predicates for x in domain.predicate_arguments[(a, p)]]
            for a in domain.action_schemas}


def max_slots(domain: Domain) -> int:
    return max([len(v) for v in lifted_slots(domain).values()] + [1])


def schema_triple_table(domain: Domain) -> torch.Tensor:
    """[S + 1, K] long: 1 + the flat triple index of slot k of schema s, 0 where schema s has no
    slot k. Row S (all zeros) is used for padded steps."""
    slots = lifted_slots(domain)
    table = torch.zeros(len(domain.action_schemas) + 1, max_slots(domain), dtype=torch.long)
    offset = 0
    for s, a in enumerate(domain.action_schemas):
        n = len(slots[a])
        table[s, :n] = torch.arange(offset + 1, offset + n + 1)
        offset += n
    return table


# ---------------------------------------------------------------------------
# Parsing of user-facing atom / action forms
# ---------------------------------------------------------------------------

def parse_atom(x) -> Atom:
    """Normalise an atom (or an action call) to (name, (arg, ...)). Also accepts planning_structs
    Action and Proposition objects, whose objects are then matched by name."""
    if isinstance(x, Action):
        return x.action_schema.name, tuple(o.name for o in x.args)
    if isinstance(x, Proposition):
        return x.predicate.name, tuple(o.name for o in x.args)
    if isinstance(x, str):
        parts = x.strip().strip("()").split()
        if not parts:
            raise ValueError(f"empty atom {x!r}")
        return parts[0], tuple(parts[1:])
    x = tuple(x)
    if not x or not isinstance(x[0], str):
        raise ValueError(f"cannot parse atom {x!r}")
    if len(x) == 2 and isinstance(x[1], (tuple, list)):
        return x[0], tuple(x[1])
    return x[0], tuple(x[1:])


def _is_index(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _check_unique(order: Sequence, what: str) -> List[Atom]:
    parsed = [parse_atom(a) for a in order]
    seen: Dict[Atom, int] = {}
    for i, a in enumerate(parsed):
        if a in seen:
            raise ValueError(f"{what} lists {atom_str(a)} twice (positions {seen[a]} and {i})")
        seen[a] = i
    return parsed


def atom_str(atom: Atom) -> str:
    name, args = atom
    return f"({name}{''.join(' ' + a for a in args)})"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GroundedAction(NamedTuple):
    """An action call ready for batching: its schema index and its slots' proposition indices."""
    schema: int
    props: Tuple[int, ...]


@dataclass
class PropositionTable:
    """Proposition index of one object-set signature: (predicate index, object positions) -> index,
    in Instance.propositions order."""
    signature: Signature
    num_props: int
    index: Dict[Tuple[int, Tuple[int, ...]], int]


@dataclass
class GroundedBatch:
    """Index tensors for grounding a padded batch of transitions, with its masks.

    schema    [B, T]     schema index of each transition (S = padding)
    props     [B, T, K]  proposition index of each slot, in the trace's own proposition order
    slot_mask [B, T, K]  True for slots the transition's schema actually has
    prop_mask [B, P]     True for each trace's real propositions
    step_mask [B, T]     True for real transitions (steps are left-aligned in each trace)
    """
    schema: torch.Tensor
    props: torch.Tensor
    slot_mask: torch.Tensor
    prop_mask: torch.Tensor
    step_mask: torch.Tensor

    @property
    def num_props_max(self) -> int:
        return self.prop_mask.shape[1]

    @property
    def num_props(self) -> torch.Tensor:
        return self.prop_mask.sum(1)

    @property
    def num_steps(self) -> torch.Tensor:
        return self.step_mask.sum(1)

    @property
    def num_transitions(self) -> int:
        return int(self.step_mask.sum())

    def touched(self) -> torch.Tensor:
        """[B, T, P] bool: propositions some slot of the transition's action grounds to."""
        hits = torch.zeros(*self.schema.shape, self.num_props_max, dtype=torch.long, device=self.props.device)
        hits.scatter_add_(-1, self.props, self.slot_mask.long())
        return hits > 0

    def to(self, device, non_blocking: bool = False) -> "GroundedBatch":
        return GroundedBatch(*(getattr(self, f.name).to(device, non_blocking=non_blocking) for f in fields(self)))

    def pin_memory(self) -> "GroundedBatch":
        """Lets DataLoader(pin_memory=True) handle GroundedBatch."""
        return GroundedBatch(*(getattr(self, f.name).pin_memory() for f in fields(self)))


class PropositionReorderer:
    """Reorders the last dimension of tensors between a data's own proposition order and the
    package's (Instance) order; made by Grounder.reorderer.

        reorder = grounder.reorderer(objects, proposition_order)
        states = reorder(network_output)      # [..., P_data] -> [..., P], differentiable
        back = reorder.inverse(states)        # [..., P] -> [..., P_data]

    `index[j]` is the data column holding package proposition j, or P_data where the data does not
    provide it (filled with `fill_value`)."""

    def __init__(self, index: torch.Tensor, num_data: int, fill_value: float = 0.0):
        self.index = index
        self.num_data = num_data
        self.fill_value = fill_value
        inverse = torch.full((num_data,), len(index), dtype=torch.long)
        provided = index < num_data
        inverse[index[provided]] = torch.arange(len(index))[provided]
        self.inverse_index = inverse

    @property
    def num_props(self) -> int:
        return len(self.index)

    @staticmethod
    def _gather(x: torch.Tensor, index: torch.Tensor, fill_value: float) -> torch.Tensor:
        padded = torch.cat([x, x.new_full((*x.shape[:-1], 1), fill_value)], dim=-1)
        return padded.index_select(-1, index.to(x.device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.num_data:
            raise ValueError(f"expected {self.num_data} entries in the last dimension (the proposition order), "
                             f"got {x.shape[-1]}")
        return self._gather(x, self.index, self.fill_value)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Back to the data's order; data entries with no proposition in this object set get fill_value."""
        if y.shape[-1] != self.num_props:
            raise ValueError(f"expected {self.num_props} entries in the last dimension, got {y.shape[-1]}")
        return self._gather(y, self.inverse_index, self.fill_value)


# ---------------------------------------------------------------------------
# Grounder
# ---------------------------------------------------------------------------

class Grounder:
    """Grounds action calls and states of traces whose object sets may differ, and collates them
    into GroundedBatches. Holds no parameters and no device state, so it can live in a Dataset or a
    collate_fn (including DataLoader workers)."""

    def __init__(self, domain: Domain):
        self.domain = domain
        self.schemas: List[ActionSchema] = list(domain.action_schemas)
        self.schema_id = {a: i for i, a in enumerate(self.schemas)}
        self.slots = lifted_slots(domain)
        self.k_max = max_slots(domain)
        self.pad_schema = len(self.schemas)
        self._pred_idx = {p.name: i for i, p in enumerate(domain.predicates)}
        self._slot_preds = {a: [(self._pred_idx[p.name], x) for p, x in self.slots[a]] for a in self.schemas}
        self._prop_tables: Dict[Signature, PropositionTable] = {}
        self._instances: Dict[Signature, Instance] = {}
        self._dense: Dict[Signature, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # ----- object sets -----

    @staticmethod
    def signature(objects: ObjectsLike) -> Signature:
        """The sequence of object types, which is all grounding depends on."""
        if isinstance(objects, Instance):
            return tuple(o.type.name for o in objects.objects)
        return tuple(t for _, t in objects)

    @staticmethod
    def _names(objects: ObjectsLike) -> List[str]:
        return [o.name for o in objects.objects] if isinstance(objects, Instance) else [n for n, _ in objects]

    def propositions(self, objects: ObjectsLike) -> PropositionTable:
        """The proposition index of this object set (cached per signature)."""
        sig = self.signature(objects)
        tab = self._prop_tables.get(sig)
        if tab is None:
            types = [self.domain.get_type(t) for t in sig]
            for t, name in zip(types, sig):
                if t is None:
                    raise ValueError(f"unknown object type '{name}'")
            inst = self.instance(objects)   # proposition order is defined by Instance
            pos = {o: i for i, o in enumerate(inst.objects)}
            index = {(self._pred_idx[p.predicate.name], tuple(pos[o] for o in p.args)): i
                     for i, p in enumerate(inst.propositions)}
            tab = self._prop_tables[sig] = PropositionTable(sig, len(index), index)
        return tab

    def instance(self, objects: ObjectsLike) -> Instance:
        """The Instance for this object set's signature, cached. Its object names may differ from
        yours (o0, o1, ...) but its proposition and action order is the same."""
        sig = self.signature(objects)
        if sig not in self._instances:
            self._instances[sig] = objects if isinstance(objects, Instance) else \
                Instance(self.domain, [(f"o{i}", t) for i, t in enumerate(sig)])
        return self._instances[sig]

    # ----- actions -----

    def ground_action(self, objects: ObjectsLike, action, positions: Optional[Dict[str, int]] = None
                      ) -> Optional[GroundedAction]:
        """Ground one action call. Returns None when the call repeats an object: ROSAME's groundings
        bind distinct objects to distinct parameters, so such a transition cannot be represented and
        is ignored (masked) in training."""
        name, args = parse_atom(action)
        a = self.domain.get_action_schema(name)
        if a is None:
            raise ValueError(f"unknown action schema '{name}'")
        if len(args) != a.arity:
            raise ValueError(f"{atom_str((name, args))}: {a.name} takes {a.arity} arguments")
        pos = positions if positions is not None else {n: i for i, n in enumerate(self._names(objects))}
        try:
            idx = tuple(pos[o] for o in args)
        except KeyError as e:
            raise ValueError(f"{atom_str((name, args))}: unknown object {e.args[0]!r}") from None
        if len(set(idx)) < len(idx):
            return None
        sig = self.signature(objects)
        for k, (i, t) in enumerate(zip(idx, a.types)):
            if not self.domain.get_type(sig[i]).is_child_of(t):
                raise ValueError(f"{atom_str((name, args))}: argument {k + 1} is not a {t.name}")
        index = self.propositions(objects).index
        return GroundedAction(self.schema_id[a], tuple(index[(pi, tuple(idx[xi - 1] for xi in x))]
                                                       for pi, x in self._slot_preds[a]))

    def action_index(self, objects: ObjectsLike, action) -> int:
        """Index of an action call in Instance(domain, objects).actions."""
        name, args = parse_atom(action)
        names = self._names(objects)
        inst = self.instance(objects)
        a = self.domain.get_action_schema(name)
        act = inst.get_action(a, [inst.objects[names.index(o)] for o in args]) if a is not None else None
        if act is None:
            raise ValueError(f"{atom_str((name, args))} is not a grounded action of this object set")
        return inst.actions.index(act)

    def ground_index(self, objects: ObjectsLike, index: int) -> GroundedAction:
        """The action at `index` in Instance(domain, objects).actions, grounded."""
        schema, props, slot_mask = self._dense_table(objects)
        if not 0 <= index < len(schema):
            raise IndexError(f"action index {index} is outside the {len(schema)} grounded actions")
        k = int(slot_mask[index].sum())
        return GroundedAction(int(schema[index]), tuple(props[index, :k].tolist()))

    def action_index_map(self, objects: ObjectsLike, action_order: Sequence) -> torch.Tensor:
        """For data whose integer actions index its own list `action_order`: a LongTensor mapping
        each of those indices to the index in Instance(domain, objects).actions, or -1 for actions
        that repeat an object (which ROSAME ignores). Duplicates and unknown actions are errors."""
        out = []
        for k, (name, args) in enumerate(_check_unique(action_order, "action_order")):
            if len(set(args)) < len(args):
                out.append(-1)
                continue
            try:
                out.append(self.action_index(objects, (name, args)))
            except ValueError as e:
                raise ValueError(f"action_order[{k}] {atom_str((name, args))}: {e}") from None
        return torch.tensor(out, dtype=torch.long)

    def reorderer(self, objects: ObjectsLike, proposition_order: Sequence, *, missing: str = "error",
                  on_unknown: str = "warn") -> PropositionReorderer:
        """A PropositionReorderer from data whose vector entries follow `proposition_order` (atoms in
        any accepted form, or Proposition objects) to this object set's proposition order.

        missing     propositions of the object set that the order does not name: "error" (default)
                    or "zero" (filled with 0)
        on_unknown  atoms in the order with no proposition here (unknown names, wrong types, repeated
                    objects): "warn" (default), "ignore" or "raise"; their entries are dropped
        Duplicated atoms are always an error."""
        if missing not in ("error", "zero"):
            raise ValueError("missing must be 'error' or 'zero'")
        tab = self.propositions(objects)
        names = self._names(objects)
        pos = {n: i for i, n in enumerate(names)}
        index = torch.full((tab.num_props,), len(proposition_order), dtype=torch.long)
        found, unknown = 0, []
        for i, (name, args) in enumerate(_check_unique(proposition_order, "proposition_order")):
            j = tab.index.get((self._pred_idx.get(name, -1), tuple(pos.get(o, -1) for o in args)))
            if j is None:
                unknown.append(atom_str((name, args)))
                continue
            index[j] = i
            found += 1
        if unknown:
            msg = (f"{len(unknown)} atom(s) in proposition_order have no proposition in this object set, "
                   f"e.g. {unknown[0]} (repeated objects, unknown names or wrong types); their entries are dropped")
            if on_unknown == "raise":
                raise ValueError(msg)
            if on_unknown == "warn":
                warnings.warn(msg, stacklevel=2)
        if found < tab.num_props and missing == "error":
            inst = self.instance(objects)
            canon = {o: names[i] for i, o in enumerate(inst.objects)}
            j = int(torch.nonzero(index == len(proposition_order))[0])
            p = inst.propositions[j]
            example = atom_str((p.predicate.name, tuple(canon[o] for o in p.args)))
            raise ValueError(f"{tab.num_props - found} proposition(s) of this object set are not in "
                             f"proposition_order, e.g. {example}; pass missing='zero' to fill them with 0")
        return PropositionReorderer(index, len(proposition_order))

    def _dense_table(self, objects: ObjectsLike):
        """All grounded actions of an object set (needs Instance.actions; small domains only)."""
        sig = self.signature(objects)
        if sig not in self._dense:
            inst = self.instance(objects)
            pos = {o.name: i for i, o in enumerate(inst.objects)}
            A = inst.num_actions
            schema = torch.empty(A, dtype=torch.long)
            props = torch.zeros(A, self.k_max, dtype=torch.long)
            slot_mask = torch.zeros(A, self.k_max, dtype=torch.bool)
            for i, act in enumerate(inst.actions):
                g = self.ground_action(inst, (act.action_schema.name, [o.name for o in act.args]), pos)
                schema[i] = g.schema
                props[i, :len(g.props)] = torch.tensor(g.props, dtype=torch.long)
                slot_mask[i, :len(g.props)] = True
            self._dense[sig] = (schema, props, slot_mask)
        return self._dense[sig]

    # ----- states -----

    def encode_state(self, objects: ObjectsLike, state, positions: Optional[Dict[str, int]] = None,
                     on_unknown: str = "warn") -> torch.Tensor:
        """A state as a float vector [P] in proposition order. `state` is an iterable of true atoms,
        a dict {atom: probability}, or already a tensor [P]. Atoms with no proposition (unknown
        predicate or object, repeated objects, wrong types) are handled per on_unknown: "warn",
        "ignore" or "raise"."""
        tab = self.propositions(objects)
        if isinstance(state, torch.Tensor):
            if state.shape != (tab.num_props,):
                raise ValueError(f"state tensor has shape {tuple(state.shape)}, expected ({tab.num_props},)")
            return state.float()
        pos = positions if positions is not None else {n: i for i, n in enumerate(self._names(objects))}
        items = state.items() if isinstance(state, dict) else ((a, 1.0) for a in state)
        s = torch.zeros(tab.num_props)
        unknown = []
        for atom, prob in items:
            name, args = parse_atom(atom)
            j = tab.index.get((self._pred_idx.get(name, -1), tuple(pos.get(o, -1) for o in args)))
            if j is None:
                unknown.append(atom_str((name, args)))
            else:
                s[j] = float(prob)
        if unknown:
            msg = f"{len(unknown)} atom(s) have no proposition in this object set, e.g. {unknown[0]}"
            if on_unknown == "raise":
                raise ValueError(msg)
            if on_unknown == "warn":
                warnings.warn(msg + " (repeated objects, unknown names or wrong types); they are ignored", stacklevel=2)
        return s

    def decode_state(self, objects: ObjectsLike, state: torch.Tensor, threshold: float = 0.5) -> List[Atom]:
        """True atoms of a state vector (entries > threshold), with your object names."""
        inst = self.instance(objects)
        names = self._names(objects)
        pos = {o: i for i, o in enumerate(inst.objects)}
        return [(p.predicate.name, tuple(names[pos[o]] for o in p.args))
                for p, v in zip(inst.propositions, state.tolist()) if v > threshold]

    # ----- batching -----

    def collate(self, objects_list: Sequence[ObjectsLike], actions_list: Sequence[Sequence],
                action_orders: Optional[Sequence[Optional[Sequence]]] = None) -> GroundedBatch:
        """One GroundedBatch from one action sequence per trace. Each action is an action call
        (see module docstring), an Action object, a GroundedAction, an integer, or None (or a
        negative integer) for a step to mask. Integers index `action_orders[b]` when given for that
        trace, otherwise Instance(domain, objects).actions. Sequences may differ in length."""
        B = len(objects_list)
        T = max([len(a) for a in actions_list] + [0])
        schema = torch.full((B, T), self.pad_schema, dtype=torch.long)
        props = torch.zeros(B, T, self.k_max, dtype=torch.long)
        slot_mask = torch.zeros(B, T, self.k_max, dtype=torch.bool)
        step_mask = torch.zeros(B, T, dtype=torch.bool)
        for b, (objects, actions) in enumerate(zip(objects_list, actions_list)):
            pos = {n: i for i, n in enumerate(self._names(objects))}
            order = action_orders[b] if action_orders is not None else None
            if order is not None:
                _check_unique(order, "action_order")
            if isinstance(actions, torch.Tensor):
                actions = actions.tolist()
            for t, act in enumerate(actions):
                if act is None or (_is_index(act) and act < 0):
                    continue
                if _is_index(act) and order is not None:
                    if act >= len(order):
                        raise IndexError(f"trace {b}: action {act} is outside its action_order ({len(order)} actions)")
                    act = order[act]
                if _is_index(act):
                    dense = self._dense_table(objects)
                    if act >= len(dense[0]):
                        raise IndexError(f"trace {b}: action index {act} >= {len(dense[0])} grounded actions")
                    schema[b, t], props[b, t], slot_mask[b, t] = dense[0][act], dense[1][act], dense[2][act]
                else:
                    g = act if isinstance(act, GroundedAction) else self.ground_action(objects, act, pos)
                    if g is None:
                        continue
                    schema[b, t] = g.schema
                    props[b, t, :len(g.props)] = torch.tensor(g.props, dtype=torch.long)
                    slot_mask[b, t, :len(g.props)] = True
                step_mask[b, t] = True
        return GroundedBatch(schema, props, slot_mask, self._prop_mask(objects_list), step_mask)

    def collate_all_actions(self, objects_list: Sequence[ObjectsLike]) -> GroundedBatch:
        """Every grounded action of each object set, e.g. to score all candidate actions when the
        actions are not observed. The step dimension enumerates actions ([B, A_max])."""
        tables = [self._dense_table(o) for o in objects_list]
        B, A = len(tables), max([len(t[0]) for t in tables] + [0])
        schema = torch.full((B, A), self.pad_schema, dtype=torch.long)
        props = torch.zeros(B, A, self.k_max, dtype=torch.long)
        slot_mask = torch.zeros(B, A, self.k_max, dtype=torch.bool)
        step_mask = torch.zeros(B, A, dtype=torch.bool)
        for b, (s, p, m) in enumerate(tables):
            schema[b, :len(s)], props[b, :len(s)], slot_mask[b, :len(s)], step_mask[b, :len(s)] = s, p, m, True
        return GroundedBatch(schema, props, slot_mask, self._prop_mask(objects_list), step_mask)

    def _prop_mask(self, objects_list: Sequence[ObjectsLike]) -> torch.Tensor:
        n = torch.tensor([self.propositions(o).num_props for o in objects_list], dtype=torch.long)
        return torch.arange(int(n.max()) if len(n) else 0)[None, :] < n[:, None]
