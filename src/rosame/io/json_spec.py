"""
JSON input and output.

Domain specification (read_domain_json / write_domain_json):

    {"name": "switches",
     "types":          [{"name": "room", "parent": null}, {"name": "switch", "parent": "device"}, ...],
     "predicates":     [{"name": "in", "types": ["device", "room"]}, ...],
     "action_schemas": [{"name": "turn-on", "types": ["switch", "room"]}, ...]}

("actions" is accepted in place of "action_schemas". Types whose parent is null are placed under
"object".)

Objects (read_objects_json / write_objects_json):   {"objects": [{"name": "r1", "type": "room"}, ...]}

Traces, one JSON object per line (read_traces_jsonl / write_traces_jsonl):

    {"objects": [["r1", "room"], ["r2", "room"], ["s1", "switch"]],
     "states":  [["(robot-at r1)", "(connected r1 r2)"], {"(robot-at r2)": 0.9, "(on s1)": 0.1}, ...],
     "actions": ["(move r1 r2)", ...]}

A state is a list of true atoms or an object {atom: probability}; an action may be null (unknown).

Learned model (write_action_model_json / read_action_model_json): per action schema, its
parameter types and its precondition / add / delete literals as [predicate, [parameter positions]].
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch

from planning_structs.action_model import ActionModel
from planning_structs.domain import Domain

Source = Union[str, Path]


def _write(text: str, path: Optional[Source]) -> str:
    if path is not None:
        Path(path).write_text(text)
    return text


# ----- domains -----

def read_domain_json(path: Source) -> Domain:
    spec = json.loads(Path(path).read_text())
    actions = spec.get("action_schemas", spec.get("actions"))
    if not isinstance(spec.get("types"), list) or not isinstance(spec.get("predicates"), list) or not isinstance(actions, list):
        raise ValueError("domain JSON needs lists 'types', 'predicates' and 'action_schemas'")
    return Domain([(t["name"], t.get("parent")) for t in spec["types"]],
                  [(p["name"], p.get("types", [])) for p in spec["predicates"]],
                  [(a["name"], a.get("types", [])) for a in actions], name=spec.get("name"))


def write_domain_json(domain: Domain, path: Optional[Source] = None) -> str:
    depth = lambda t: 0 if t.parent is None else 1 + depth(t.parent)
    types = [{"name": t.name, "parent": t.parent.name if t.parent else None}
             for t in sorted(domain.types_set, key=lambda t: (depth(t), t.name))]
    payload = {"name": domain.name, "types": types,
               "predicates": [{"name": p.name, "types": [t.name for t in p.types]} for p in domain.predicates],
               "action_schemas": [{"name": a.name, "types": [t.name for t in a.types]} for a in domain.action_schemas]}
    return _write(json.dumps(payload, indent=2) + "\n", path)


# ----- objects -----

def read_objects_json(path: Source) -> List[Tuple[str, str]]:
    spec = json.loads(Path(path).read_text())
    return [(o["name"], o["type"]) for o in spec["objects"]]


def write_objects_json(objects: Sequence[Tuple[str, str]], path: Optional[Source] = None) -> str:
    return _write(json.dumps({"objects": [{"name": n, "type": t} for n, t in objects]}, indent=2) + "\n", path)


# ----- learned models -----

def write_action_model_json(model, path: Optional[Source] = None) -> str:
    """A learned model (ROSAME or ActionModel) as JSON."""
    am: ActionModel = model.action_model() if callable(getattr(model, "action_model", None)) else model
    lits = lambda ls: [[p.name, list(x)] for p, x in ls]
    payload = {"domain": am.domain.name, "actions": [
        {"name": a.name, "parameters": [t.name for t in a.types],
         "preconditions": lits(am.schema(a.name)[0]), "add": lits(am.schema(a.name)[1]),
         "delete": lits(am.schema(a.name)[2])} for a in am.domain.action_schemas]}
    return _write(json.dumps(payload, indent=2) + "\n", path)


def read_action_model_json(path: Source, domain: Domain) -> ActionModel:
    spec = json.loads(Path(path).read_text())
    am = ActionModel(domain)
    for a in spec["actions"]:
        for key, book in (("preconditions", am.preconditions), ("add", am.add_effects), ("delete", am.del_effects)):
            for pname, x in a.get(key, []):
                p = domain.get_predicate(pname)
                if p is None:
                    raise ValueError(f"unknown predicate '{pname}'")
                book.setdefault(a["name"], []).append((p, tuple(x)))
    return am


# ----- traces -----

def read_traces_jsonl(path: Source):
    from planning_structs import Trace
    traces = []
    for k, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        try:
            traces.append(Trace(objects=[tuple(o) if isinstance(o, list) else (o["name"], o["type"]) for o in d["objects"]],
                                states=d["states"], actions=d["actions"]))
        except KeyError as e:
            raise ValueError(f"line {k + 1}: missing key {e.args[0]!r}") from None
    return traces


def write_traces_jsonl(traces, path: Source, domain: Optional[Domain] = None) -> None:
    """Write traces as JSON lines. Tensor states need `domain` (to name the propositions) and are
    written as {atom: probability} for their non-zero entries."""
    from rosame.grounding import Grounder, atom_str, parse_atom
    grounder = Grounder(domain) if domain is not None else None
    with open(path, "w") as f:
        for tr in traces:
            states = tr.states
            if isinstance(states, torch.Tensor):
                if grounder is None:
                    raise ValueError("tensor states need the domain to be written")
                inst = grounder.instance(tr.objects)
                names = [n for n, _ in tr.objects]
                pos = {o: i for i, o in enumerate(inst.objects)}
                keys = [atom_str((p.predicate.name, tuple(names[pos[o]] for o in p.args))) for p in inst.propositions]
                states = [{keys[j]: round(float(v), 6) for j, v in enumerate(s.tolist()) if v} for s in states]
            else:
                states = [{atom_str(parse_atom(a)): float(v) for a, v in s.items()} if isinstance(s, dict)
                          else [atom_str(parse_atom(a)) for a in s] for s in states]
            actions = [None if a is None else atom_str(parse_atom(a)) for a in tr.actions]
            f.write(json.dumps({"objects": [list(o) for o in tr.objects], "states": states, "actions": actions}) + "\n")
