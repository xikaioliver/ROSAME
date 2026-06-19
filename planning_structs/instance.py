from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import permutations
from planning_structs.domain import Type, Predicate, ActionSchema, Domain

# ---- Utility -------------------------------------------------

def objs2str(args: Tuple["Object", ...]) -> str:
    return "_".join(o.name for o in args) if args else ""

# ---- Core instance objects -----------------------------------

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
    args: Tuple[Object, ...]  # <-- Object tuple

    @property
    def arity(self) -> int:
        return len(self.args)

    def __str__(self) -> str:
        return self.predicate.name if not self.args else f"{self.predicate.name}_{objs2str(self.args)}"

@dataclass(eq=True, frozen=True)
class Action:
    """Grounded action with Object arguments."""
    action_schema: "ActionSchema"
    args: Tuple[Object, ...]  # <-- Object tuple

    @property
    def arity(self) -> int:
        return len(self.args)

    def __str__(self) -> str:
        return self.action_schema.name if not self.args else f"{self.action_schema.name}_{objs2str(self.args)}"

# ---- Instance ------------------------------------------------

class Instance:
    """
    Builds grounded Objects, Actions, and Propositions for a Domain.
    Uses permutations → distinct objects per grounding (matches binding enumeration).
    """
    def __init__(self, domain: Domain, objects: List[Tuple[str, str]]):
        self.domain = domain
        self.objects: List[Object] = self._build_objects(objects)
        self.actions: List[Action] = self._build_actions()
        self.propositions: List[Proposition] = self._build_propositions()
        self.actors: Dict[Proposition, List[Action]] = self.build_actors()

        # Indices for fast lookup
        self._actions_dict: Dict[Tuple[ActionSchema, Tuple[Object, ...]], Action] = {
            (a.action_schema, a.args): a for a in self.actions
        }
        self._propositions_dict: Dict[Tuple[Predicate, Tuple[Object, ...]], Proposition] = {
            (p.predicate, p.args): p for p in self.propositions
        }
        self._objects_dict: Dict[str, Object] = {
            o.name: o for o in self.objects
        }
        self.prop_index: Dict[Tuple[Predicate, Tuple[Object, ...]], int] = {
            (p.predicate, p.args): i for i, p in enumerate(self.propositions)
        }

    # ---------- Builders ----------

    def _build_objects(self, objects: List[Tuple[str, str]]) -> List[Object]:
        objs: List[Object] = []
        for name, type_name in objects:
            t = self.domain.get_type(type_name)
            if t is None:
                raise ValueError(f"Unknown type '{type_name}' for object '{name}'.")
            objs.append(Object(name=name, type=t))
        return objs

    def _type_match(self, objs: Tuple[Object, ...], ap) -> bool:
        # Each object type must be a child of the slot type.
        for i in range(len(objs)):
            if not objs[i].type.is_child_of(ap.types[i]):
                return False
        return True

    def _build_actions(self) -> List[Action]:
        acts: List[Action] = []
        for a in self.domain.action_schemas:
            for tup in permutations(self.objects, a.arity):
                if self._type_match(tup, a):
                    acts.append(Action(a, tup))
        return acts

    def _build_propositions(self) -> List[Proposition]:
        props: List[Proposition] = []
        for p in self.domain.predicates:
            for tup in permutations(self.objects, p.arity):
                if self._type_match(tup, p):
                    props.append(Proposition(p, tup))
        return props

    def build_actors(self):
        result: Dict[Proposition, List[Action]] = {}
        for p in self.propositions:
            result[p] = []
            for a in self.actions:
                belongs = True
                for i in range(0, p.predicate.arity):
                    if p.args[i] not in a.args:
                        belongs = False
                        break
                if belongs:
                    result[p].append(a)
        return result

    # ---------- Lookups ----------

    def get_action(self, action_schema: ActionSchema, args: List[Object] | Tuple[Object, ...]) -> Optional[Action]:
        key = (action_schema, tuple(args))
        return self._actions_dict.get(key)

    def get_permuted_action(self, action: Action, name_ext_int=None, args_ext_int=None):
        action_schema_int = action.action_schema if name_ext_int is None else self.domain.get_action_schema(name_ext_int[action.action_schema.name])
        if args_ext_int is not None:
            args_mapping = args_ext_int[action.action_schema.name]
            args_int = [action.args[i - 1] for i in args_mapping]
        else:
            args_int = action.args
        return self.get_action(action_schema_int, args_int)

    def get_proposition(self, predicate: Predicate, args: List[Object] | Tuple[Object, ...]) -> Optional[Proposition]:
        key = (predicate, tuple(args))
        return self._propositions_dict.get(key)

    def get_object(self, name: str) -> Optional[Object]:
        return self._objects_dict.get(name)

    # ---------- Convenience ----------

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    @property
    def num_propositions(self) -> int:
        return len(self.propositions)
