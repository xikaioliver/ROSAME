"""
A symbolic (discrete) lifted STRIPS action model over a Domain.

Each literal of an action schema is a (predicate, binding) pair, where the binding lists, for each
predicate position, the 1-based position of the action parameter it refers to. For example, in
`turn-on(?s, ?r)` the literal `(in ?s ?r)` is (in, (1, 2)).

ROSAME learns a probability over four roles for every literal slot (see rosame.model.ROLES);
ActionModel holds the discretised result, independent of torch, so it can be inspected, compared or
written out with rosame.io.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from rosame.planning_structs.domain import Domain, Predicate

Literal = Tuple[Predicate, Tuple[int, ...]]


@dataclass(eq=False)
class ActionModel:
    domain: Domain
    preconditions: Dict[str, List[Literal]] = field(default_factory=dict)
    add_effects: Dict[str, List[Literal]] = field(default_factory=dict)
    del_effects: Dict[str, List[Literal]] = field(default_factory=dict)

    def schema(self, name: str) -> Tuple[List[Literal], List[Literal], List[Literal]]:
        """(preconditions, add effects, delete effects) of one action schema."""
        return (self.preconditions.get(name, []), self.add_effects.get(name, []), self.del_effects.get(name, []))

    @staticmethod
    def literal_str(lit: Literal, parameter_names=None) -> str:
        p, x = lit
        names = parameter_names or [f"?x{i}" for i in range(1, max(x, default=0) + 1)]
        return f"({p.name}{''.join(' ' + names[i - 1] for i in x)})"

    def _key(self):
        schemas = {a.name for a in self.domain.action_schemas}
        return {n: tuple(frozenset((p.name, x) for p, x in lits) for lits in self.schema(n)) for n in schemas}

    def __eq__(self, other) -> bool:
        """Same literals per action schema (by name), regardless of order."""
        return isinstance(other, ActionModel) and self._key() == other._key()

    def diff(self, other: "ActionModel") -> List[str]:
        """Readable differences from another model, e.g. a reference: one line per literal that is
        in only one of the two (empty if they are equal)."""
        mine, theirs, out = self._key(), other._key(), []
        for name in sorted(set(mine) | set(theirs)):
            for part, a, b in zip(("precondition", "add effect", "delete effect"),
                                  mine.get(name, (frozenset(),) * 3), theirs.get(name, (frozenset(),) * 3)):
                for p, x in sorted(a - b):
                    out.append(f"{name}: {part} ({p}{''.join(f' ?x{i}' for i in x)}) only in this model")
                for p, x in sorted(b - a):
                    out.append(f"{name}: {part} ({p}{''.join(f' ?x{i}' for i in x)}) only in the other model")
        return out

    def __str__(self) -> str:
        lines = []
        for a in self.domain.action_schemas:
            names = [f"?x{i}" for i in range(1, a.arity + 1)]
            pre, add, dele = self.schema(a.name)
            fmt = lambda lits: " ".join(self.literal_str(l, names) for l in lits) or "-"
            lines.append(f"{a.name}({', '.join(names)})\n  pre: {fmt(pre)}\n  add: {fmt(add)}\n  del: {fmt(dele)}")
        return "\n".join(lines)
