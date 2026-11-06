"""
PDDL input and output for the STRIPS-with-typing fragment ROSAME learns.

    pd = read_pddl_domain("domain.pddl")          # PDDLDomain: .domain is a planning_structs.Domain
    problem = read_pddl_problem("p01.pddl", pd)   # objects (incl. domain constants), init, goal
    text = write_pddl(model, pddl_domain=pd)      # learned model, with pd's parameter names and header

Supported when reading: :types (single inheritance), :constants, :predicates, and actions whose
preconditions/effects are conjunctions of atoms and negated atoms (anything else in a precondition
or effect is reported in PDDLDomain.unsupported, and never needed for learning, which only uses the
signatures). `either` types, conditional effects, quantifiers and numeric fluents are not supported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from planning_structs.action_model import ActionModel
from planning_structs.domain import Domain

Atom = Tuple[str, Tuple[str, ...]]
Source = Union[str, Path]


# ---------------------------------------------------------------------------
# s-expressions
# ---------------------------------------------------------------------------

def _text(source: Source) -> str:
    """PDDL text from a path or from the text itself."""
    if isinstance(source, str) and "(define" in source.lower():
        return source
    return Path(source).read_text()


def parse_sexp(text: str):
    text = re.sub(r";[^\n]*", "", text).lower()
    stack: list = [[]]
    for tok in re.findall(r"\(|\)|[^\s()]+", text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced parentheses: unexpected ')'")
            last = stack.pop()
            stack[-1].append(last)
        else:
            stack[-1].append(tok)
    if len(stack) != 1 or not stack[0]:
        raise ValueError("unbalanced parentheses or empty input")
    return stack[0][0]


def _typed_list(items) -> List[Tuple[str, str]]:
    out, pending, i = [], [], 0
    while i < len(items):
        if items[i] == "-":
            if i + 1 >= len(items) or not isinstance(items[i + 1], str):
                raise ValueError(f"unsupported type expression after '-': {items[i + 1:i + 2]} ('either' is not supported)")
            out += [(n, items[i + 1]) for n in pending]
            pending, i = [], i + 2
        else:
            pending.append(items[i])
            i += 1
    return out + [(n, "object") for n in pending]


def _sections(tree, key):
    return [x for x in tree if isinstance(x, list) and x and x[0] == key]


def _conjuncts(expr) -> List:
    if not expr:
        return []
    if expr[0] == "and":
        return expr[1:]
    return [expr]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@dataclass
class PDDLDomain:
    """A PDDL domain: the typed signatures as a Domain, plus what is needed to write it back."""
    domain: Domain
    requirements: List[str]
    types: List[Tuple[str, Optional[str]]]
    constants: List[Tuple[str, str]]
    predicate_parameters: Dict[str, List[str]]
    action_parameters: Dict[str, List[str]]
    action_model: ActionModel
    unsupported: List[str] = field(default_factory=list)


@dataclass
class PDDLProblem:
    name: str
    objects: List[Tuple[str, str]]     # problem objects followed by the domain's constants
    init: List[Atom]
    goal: List[Atom]                   # positive goal atoms (negated ones are listed in unsupported)
    unsupported: List[str] = field(default_factory=list)


def read_pddl_domain(source: Source) -> PDDLDomain:
    """Read a PDDL domain from a file path or a string."""
    tree = parse_sexp(_text(source))
    if tree[0] != "define":
        raise ValueError("not a PDDL definition")
    name = next(x[1] for x in tree if isinstance(x, list) and x and x[0] == "domain")
    req = _sections(tree, ":requirements")
    types = [(n, None if t == "object" else t) for n, t in _typed_list(_sections(tree, ":types")[0][1:])
             if n != "object"] if _sections(tree, ":types") else []
    constants = _typed_list(_sections(tree, ":constants")[0][1:]) if _sections(tree, ":constants") else []
    preds = [(p[0], _typed_list(p[1:])) for p in _sections(tree, ":predicates")[0][1:]] if _sections(tree, ":predicates") else []
    actions = []
    for a in _sections(tree, ":action"):
        params = a[a.index(":parameters") + 1] if ":parameters" in a else []
        actions.append((a[1], _typed_list(params), a))
    domain = Domain(types, [(n, [t for _, t in ps]) for n, ps in preds],
                    [(n, [t for _, t in ps]) for n, ps, _ in actions], name=name)
    model, unsupported = ActionModel(domain), []
    for n, params, a in actions:
        pos = {v: i + 1 for i, (v, _) in enumerate(params)}
        for key, target in ((":precondition", "pre"), (":effect", "eff")):
            if key not in a:
                continue
            for lit in _conjuncts(a[a.index(key) + 1]):
                neg = lit[0] == "not"
                atom = lit[1] if neg else lit
                p = domain.get_predicate(atom[0]) if isinstance(atom, list) and atom else None
                x = tuple(pos.get(v, 0) for v in atom[1:]) if p is not None else ()
                if p is None or 0 in x or len(set(x)) < len(x) or (neg and target == "pre"):
                    unsupported.append(f"{n} {key[1:]}: {_show(lit)}")
                    continue
                book = model.preconditions if target == "pre" else (model.del_effects if neg else model.add_effects)
                book.setdefault(n, []).append((p, x))
    return PDDLDomain(domain, req[0][1:] if req else [":strips", ":typing"], types, constants,
                      {n: [v for v, _ in ps] for n, ps in preds}, {n: [v for v, _ in ps] for n, ps, _ in actions},
                      model, unsupported)


def read_pddl_problem(source: Source, pddl_domain: Optional[PDDLDomain] = None) -> PDDLProblem:
    """Read a PDDL problem from a file path or a string. With pddl_domain, its constants are appended
    to the objects (problems use constants without declaring them)."""
    tree = parse_sexp(_text(source))
    name = next((x[1] for x in tree if isinstance(x, list) and x and x[0] == "problem"), "")
    objects = _typed_list(_sections(tree, ":objects")[0][1:]) if _sections(tree, ":objects") else []
    if pddl_domain is not None:
        declared = {n for n, _ in objects}
        objects += [(n, t) for n, t in pddl_domain.constants if n not in declared]
    init_s = _sections(tree, ":init")
    init = [(a[0], tuple(a[1:])) for a in (init_s[0][1:] if init_s else []) if isinstance(a, list) and a and a[0] != "="]
    goal, unsupported = [], []
    goal_s = _sections(tree, ":goal")
    for lit in _conjuncts(goal_s[0][1] if goal_s and len(goal_s[0]) > 1 else []):
        if isinstance(lit, list) and lit and lit[0] not in ("not", "or", "forall", "exists", "imply"):
            goal.append((lit[0], tuple(lit[1:])))
        else:
            unsupported.append(f"goal: {_show(lit)}")
    return PDDLProblem(name, objects, init, goal, unsupported)


def _show(expr) -> str:
    return expr if isinstance(expr, str) else "(" + " ".join(_show(e) for e in expr) + ")"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _default_names(n: int) -> List[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return [f"?{letters[i]}" if i < 26 else f"?v{i}" for i in range(n)]


def write_pddl(model, path: Optional[Source] = None, pddl_domain: Optional[PDDLDomain] = None,
               domain_name: Optional[str] = None) -> str:
    """A learned model (ROSAME or ActionModel) as a PDDL domain string, also written to `path` if
    given. With pddl_domain (e.g. the domain file you learned from), its requirements, types,
    constants and parameter names are kept, so the output lines up with the original file."""
    am: ActionModel = model.action_model() if hasattr(model, "action_model") and callable(model.action_model) else model
    domain = am.domain
    name = domain_name or (domain.name if domain.name else "learned")

    if pddl_domain is not None:
        requirements = pddl_domain.requirements
        type_pairs = [(n, p or "object") for n, p in pddl_domain.types]
        constants = pddl_domain.constants
        pred_params = pddl_domain.predicate_parameters
        act_params = pddl_domain.action_parameters
    else:
        requirements = [":strips", ":typing"]
        depth = lambda t: 0 if t.parent is None else 1 + depth(t.parent)
        type_pairs = [(t.name, t.parent.name) for t in sorted(domain.types_set, key=lambda t: (depth(t), t.name))
                      if t.parent is not None]
        constants = []
        pred_params = {p.name: _default_names(p.arity) for p in domain.predicates}
        act_params = {a.name: _default_names(a.arity) for a in domain.action_schemas}

    by_parent: Dict[str, List[str]] = {}
    for n, p in type_pairs:
        by_parent.setdefault(p, []).append(n)
    lines = [f"(define (domain {name})", f"  (:requirements {' '.join(requirements)})"]
    if by_parent:
        lines.append("  (:types " + " ".join(f"{' '.join(ch)} - {p}" for p, ch in by_parent.items()) + ")")
    if constants:
        lines.append("  (:constants " + " ".join(f"{n} - {t}" for n, t in constants) + ")")
    preds = [f"({p.name}{''.join(f' {v} - {t.name}' for v, t in zip(pred_params[p.name], p.types))})"
             for p in domain.predicates]
    lines.append("  (:predicates\n    " + "\n    ".join(preds) + ")")
    for a in domain.action_schemas:
        names = act_params[a.name]
        pre, add, dele = am.schema(a.name)
        fmt = lambda lit: am.literal_str(lit, names)
        effects = [fmt(l) for l in add] + [f"(not {fmt(l)})" for l in dele]
        lines.append(f"  (:action {a.name}\n"
                     f"    :parameters ({' '.join(f'{v} - {t.name}' for v, t in zip(names, a.types))})\n"
                     f"    :precondition (and {' '.join(fmt(l) for l in pre)})\n"
                     f"    :effect (and {' '.join(effects)}))")
    text = "\n".join(lines) + "\n)\n"
    if path is not None:
        Path(path).write_text(text)
    return text
