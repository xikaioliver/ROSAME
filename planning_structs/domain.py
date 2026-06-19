from __future__ import annotations
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Set, Tuple

@dataclass(eq=True, frozen=True)
class Type:
    name: str
    parent: Optional["Type"] = field(default=None, compare=False)

    def is_child_of(self, ancestor: "Type") -> bool:
        """Returns True iff self is the same as ancestor or a descendant."""
        t = self
        while t is not None:
            if t.name == ancestor.name:
                return True
            t = object.__getattribute__(t, "parent")  # avoid dataclass frozen recursion
        return False

    def __str__(self):
        return self.name


@dataclass(eq=True, frozen=True)
class Predicate:
    name: str
    arity: int
    types: Tuple[Type, ...]  # length = arity (unpadded)

    def __str__(self):
        return f"{self.name}([{', '.join([t.name for t in self.types])}])"


@dataclass(eq=True, frozen=True)
class ActionSchema:
    name: str
    arity: int
    types: Tuple[Type, ...]  # length = arity (unpadded)

    def __str__(self):
        return f"{self.name}([{', '.join([t.name for t in self.types])}])"


class Domain:
    """
    Domain(types, predicates, action_schemas)
    - types: List[Tuple[name, parent_name_or_None]]
    - predicates: List[Tuple[name, List[type_names]]]
    - action_schemas: List[Tuple[name, List[type_names]]]
    """
    def __init__(self, types, predicates, action_schemas, name: str | None = None):
        # store name if provided
        self.name = name

        self.types_set, self._types_dict, self._edges = self.build_types(types)
        self.predicates: List[Predicate] = self.build_predicates(predicates)
        self.action_schemas: List[ActionSchema] = self.build_action_schemas(action_schemas)

        # arity stats (for padding)
        self.max_pred_arity = max((p.arity for p in self.predicates), default=0)
        self.max_action_arity = max((a.arity for a in self.action_schemas), default=0)

        # all valid bindings (padded to max_pred_arity with 0s)
        self.predicate_arguments: Dict[Tuple[ActionSchema, Predicate], List[Tuple[int, ...]]] = \
            self.build_predicate_arguments()

    # --------- building blocks ---------

    def build_types(self, types_spec):
        """
        types_spec: List[Tuple[name, parent_name_or_None]]
        Returns: (types_set, name->Type dict, edges[parent, child] sorted)
        Behavior:
          - Ensures a base 'object' type exists.
          - Any type declared as a root (parent=None) and not named 'object'
            is silently attached under 'object'.
          - Any parent name that appears but isn't declared is created as a root,
            then also attached under 'object'.
        """
        # 1) Collect declared relations
        name_to_parent = {n: p for n, p in types_spec}

        # 2) Ensure 'object' exists as explicit root
        if "object" not in name_to_parent:
            name_to_parent["object"] = None

        # 3) If a parent name appears that isn't declared, create it (root for now)
        for _, p in types_spec:
            if p is not None and p not in name_to_parent:
                name_to_parent[p] = None  # implicit node

        # 4) Attach all roots (except 'object') under 'object'
        #    This handles the "loosely declared several roots" case.
        for n, p in list(name_to_parent.items()):
            if p is None and n != "object":
                name_to_parent[n] = "object"

        # --- Build Type objects with cycle detection ---
        cache = {}  # name -> Type
        visiting = set()

        def build(n: str):
            if n in cache:
                return cache[n]
            if n in visiting:
                raise ValueError(f"Cycle detected in type hierarchy at '{n}'")
            visiting.add(n)
            parent_name = name_to_parent.get(n, None)
            parent_obj = build(parent_name) if parent_name is not None else None
            t = Type(name=n, parent=parent_obj)
            cache[n] = t
            visiting.remove(n)
            return t

        # Build all nodes (declared + implicit)
        for n in list(name_to_parent.keys()):
            build(n)

        # Deterministic edges: (parent, child) excluding roots
        edges = sorted((p, c) for c, p in name_to_parent.items() if p is not None)

        types_set = set(cache.values())
        return types_set, cache, edges

    def build_predicates(self, predicates_spec) -> list[Predicate]:
        preds = []
        for (name, params) in predicates_spec:
            types = tuple(self._types_dict[tn] for tn in params)
            preds.append(Predicate(name=name, arity=len(types), types=types))
        self._pred_map = {p.name: p for p in preds}
        return preds

    def build_action_schemas(self, schemas_spec) -> list[ActionSchema]:
        acts = []
        for (name, params) in schemas_spec:
            types = tuple(self._types_dict[tn] for tn in params)
            acts.append(ActionSchema(name=name, arity=len(types), types=types))
        self._act_map = {a.name: a for a in acts}
        return acts

    def _type_match(self, x_tuple: Tuple[int, ...], a: ActionSchema, p: Predicate) -> bool:
        """
        x_tuple is a length p.arity tuple of integers in 1..a.arity (positions),
        mapping predicate positions -> action positions.
        Zeros are for padded predicate positions and are handled elsewhere; here we only check 1..p.arity.
        """
        for i in range(p.arity):
            ai = x_tuple[i]
            # ai is in 1..a.arity
            if not a.types[ai - 1].is_child_of(p.types[i]):
                return False
        return True

    def build_predicate_arguments(self):
        """
        For each (a, p), enumerate all valid bindings of length == p.arity
        mapping predicate positions -> distinct action positions.
        """
        res = {}
        for a in self.action_schemas:
            for p in self.predicates:
                valid = []
                if a.arity >= p.arity:
                    for x in permutations(range(1, a.arity + 1), p.arity):
                        if self._type_match(x, a, p):
                            valid.append(x)
                res[(a, p)] = valid
        return res

    def get_type(self, type_name: str) -> Optional[Type]:
        return self._types_dict.get(type_name)

    def get_action_schema(self, act_name: str) -> Optional[ActionSchema]:
        return self._act_map.get(act_name)

    def get_permuted_action_schema(self, act_name: str, name_ext_int=None):
        act_name = act_name if name_ext_int is None else name_ext_int[act_name]
        return self.get_action_schema(act_name)

    def get_predicate(self, pred_name: str) -> Optional[Predicate]:
        return self._pred_map.get(pred_name)