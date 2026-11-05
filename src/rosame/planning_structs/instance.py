"""
Grounding of a Domain on a set of objects: objects, propositions and grounded actions.

Order conventions (relied on throughout the package):
  - propositions: for each predicate in domain order, every tuple of *distinct*, type-compatible
    objects in lexicographic order of their positions in the object list;
  - actions: the same, for each action schema in domain order.
Object names never affect these orders, only the sequence of object types does.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from rosame.planning_structs.domain import ActionSchema, Domain, Predicate, Type


def objs2str(args: Tuple["Object", ...]) -> str:
    return "_".join(o.name for o in args) if args else ""


@dataclass(eq=True, frozen=True)
class Object:
    name: str
    type: Type

    def __str__(self) -> str:
        return f"{self.name}: {self.type.name}"


@dataclass(eq=True, frozen=True)
class Proposition:
    """Grounded predicate application with Object arguments."""
    predicate: Predicate
    args: Tuple[Object, ...]

    @property
    def arity(self) -> int:
        return len(self.args)

    def __str__(self) -> str:
        return self.predicate.name if not self.args else f"{self.predicate.name}_{objs2str(self.args)}"


@dataclass(eq=True, frozen=True)
class Action:
    """Grounded action with Object arguments."""
    action_schema: ActionSchema
    args: Tuple[Object, ...]

    @property
    def arity(self) -> int:
        return len(self.args)

    def __str__(self) -> str:
        return self.action_schema.name if not self.args else f"{self.action_schema.name}_{objs2str(self.args)}"


def _groundings(objects: Sequence[Object], types: Sequence[Type]):
    """Tuples of distinct objects whose types match `types`, in lexicographic position order
    (identical to filtering itertools.permutations(objects, len(types)) by type, but only visits
    type-compatible candidates)."""
    cands = [[i for i, o in enumerate(objects) if o.type.is_child_of(t)] for t in types]
    for idx in product(*cands):
        if len(set(idx)) == len(idx):
            yield tuple(objects[i] for i in idx)


class Instance:
    """
    Objects, propositions and grounded actions of a Domain for one object set.

    Groundings use distinct objects for distinct parameters (injective bindings), matching the
    bindings enumerated by Domain. Propositions are built on construction; grounded actions are
    built on first access to `actions`, since their number grows with the action arity and can be
    very large (the batched learner does not need them, see rosame.grounding).
    """

    def __init__(self, domain: Domain, objects: Sequence[Tuple[str, str]]):
        self.domain = domain
        self.objects: List[Object] = self._build_objects(objects)
        self.propositions: List[Proposition] = [Proposition(p, args) for p in domain.predicates
                                                for args in _groundings(self.objects, p.types)]
        self.prop_index: Dict[Tuple[Predicate, Tuple[Object, ...]], int] = {
            (p.predicate, p.args): i for i, p in enumerate(self.propositions)}
        self._objects_dict: Dict[str, Object] = {o.name: o for o in self.objects}
        self._actions: Optional[List[Action]] = None
        self._actions_dict: Optional[Dict[Tuple[ActionSchema, Tuple[Object, ...]], Action]] = None

    def _build_objects(self, objects: Sequence[Tuple[str, str]]) -> List[Object]:
        objs: List[Object] = []
        for name, type_name in objects:
            t = self.domain.get_type(type_name)
            if t is None:
                raise ValueError(f"Unknown type '{type_name}' for object '{name}'.")
            objs.append(Object(name=name, type=t))
        return objs

    # ---------- grounded actions (lazy) ----------

    @property
    def actions(self) -> List[Action]:
        if self._actions is None:
            self._actions = [Action(a, args) for a in self.domain.action_schemas
                             for args in _groundings(self.objects, a.types)]
            self._actions_dict = {(a.action_schema, a.args): a for a in self._actions}
        return self._actions

    # ---------- lookups ----------

    def get_object(self, name: str) -> Optional[Object]:
        return self._objects_dict.get(name)

    def get_action(self, action_schema: ActionSchema, args: Sequence[Object]) -> Optional[Action]:
        self.actions
        return self._actions_dict.get((action_schema, tuple(args)))

    def get_proposition(self, predicate: Predicate, args: Sequence[Object]) -> Optional[Proposition]:
        i = self.prop_index.get((predicate, tuple(args)))
        return None if i is None else self.propositions[i]

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    @property
    def num_propositions(self) -> int:
        return len(self.propositions)
