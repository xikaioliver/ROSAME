"""Test-only helpers: two small domains defined in code, a reference STRIPS simulator on sets of
atoms, and naive reference implementations to check the package against.

- fixture_domain(): abstract names, typing with inheritance (B is an A), arities 0 to 3.
- switches_domain(): a robot moves around a one-way ring of rooms and turns switches on and off when
  the power is on; it has prevail conditions (connected, robot-at, in, powered), add effects, and
  delete effects.
"""
from __future__ import annotations

import random
from itertools import permutations
from typing import List, Tuple

import torch

from rosame import ActionModel, Domain, Trace


def fixture_domain() -> Domain:
    return Domain([("A", None), ("B", "A"), ("C", None)],
                  [("p0", []), ("p1", ["A"]), ("p2", ["A", "C"]), ("p3", ["B", "B", "C"])],
                  [("x", ["B", "C"]), ("y", ["A", "A"]), ("z", ["B", "B", "C"])], name="fixture")


def fixture_objects(rng: random.Random) -> List[Tuple[str, str]]:
    return ([(f"a{i}", "A") for i in range(rng.randint(1, 2))] + [(f"b{i}", "B") for i in range(rng.randint(2, 3))]
            + [(f"c{i}", "C") for i in range(rng.randint(1, 2))])


SWITCHES_PDDL = """
(define (domain switches)
  (:requirements :strips :typing)
  (:types switch - device room device)
  (:predicates (powered) (robot-at ?r - room) (in ?d - device ?r - room) (connected ?a ?b - room)
               (on ?s - switch) (off ?s - switch))
  (:action move :parameters (?from ?to - room)
    :precondition (and (robot-at ?from) (connected ?from ?to))
    :effect (and (robot-at ?to) (not (robot-at ?from))))
  (:action turn-on :parameters (?s - switch ?r - room)
    :precondition (and (powered) (robot-at ?r) (in ?s ?r) (off ?s))
    :effect (and (on ?s) (not (off ?s))))
  (:action turn-off :parameters (?s - switch ?r - room)
    :precondition (and (powered) (robot-at ?r) (in ?s ?r) (on ?s))
    :effect (and (off ?s) (not (on ?s)))))
"""

SWITCHES_MODEL = {   # schema -> (pre, add, del), literals as (predicate, parameter positions)
    "move": ([("robot-at", (1,)), ("connected", (1, 2))], [("robot-at", (2,))], [("robot-at", (1,))]),
    "turn-on": ([("powered", ()), ("robot-at", (2,)), ("in", (1, 2)), ("off", (1,))], [("on", (1,))], [("off", (1,))]),
    "turn-off": ([("powered", ()), ("robot-at", (2,)), ("in", (1, 2)), ("on", (1,))], [("off", (1,))], [("on", (1,))]),
}


def switches_domain() -> Domain:
    return Domain([("switch", "device"), ("room", None), ("device", None)],
                  [("powered", []), ("robot-at", ["room"]), ("in", ["device", "room"]), ("connected", ["room", "room"]),
                   ("on", ["switch"]), ("off", ["switch"])],
                  [("move", ["room", "room"]), ("turn-on", ["switch", "room"]), ("turn-off", ["switch", "room"])],
                  name="switches")


def reference_action_model(domain: Domain, spec=SWITCHES_MODEL) -> ActionModel:
    am = ActionModel(domain)
    for a, (pre, add, dele) in spec.items():
        for book, lits in ((am.preconditions, pre), (am.add_effects, add), (am.del_effects, dele)):
            book[a] = [(domain.get_predicate(p), x) for p, x in lits]
    return am


def simulate(domain: Domain, spec, objects, init, steps: int, rng: random.Random):
    """Random walk under a STRIPS model, on Python sets of atoms (independent of the package)."""
    states, actions, s = [set(init)], [], set(init)
    names = [n for n, _ in objects]
    types = {n: domain.get_type(t) for n, t in objects}
    for _ in range(steps):
        options = []
        for a in domain.action_schemas:
            for args in permutations(names, a.arity):
                if not all(types[o].is_child_of(t) for o, t in zip(args, a.types)):
                    continue
                pre, add, dele = spec[a.name]
                g = lambda lits: {(p, tuple(args[i - 1] for i in x)) for p, x in lits}
                if g(pre) <= s:
                    options.append((a.name, args, g(add), g(dele)))
        name, args, add, dele = rng.choice(options)
        s = (s - dele) | add
        states.append(set(s))
        actions.append((name, args))
    return states, actions


def switches_problem(rng: random.Random):
    """Objects and initial state: 3-4 rooms in a one-way ring, 1-3 switches, power on 60% of the time."""
    rooms = [f"r{i}" for i in range(rng.randint(3, 4))]
    switches = [f"s{i}" for i in range(rng.randint(1, 3))]
    objects = [(r, "room") for r in rooms] + [(s, "switch") for s in switches]
    init = {("robot-at", (rng.choice(rooms),))}
    init |= {("connected", (rooms[i], rooms[(i + 1) % len(rooms)])) for i in range(len(rooms))}
    init |= {("in", (s, rng.choice(rooms))) for s in switches}
    init |= {(rng.choice(["on", "off"]), (s,)) for s in switches}
    if rng.random() < 0.6:
        init.add(("powered", ()))
    return objects, init


def switches_traces(rng: random.Random, n: int, lengths=(3, 8)):
    d = switches_domain()
    out = []
    for _ in range(n):
        objects, init = switches_problem(rng)
        states, actions = simulate(d, SWITCHES_MODEL, objects, init, rng.randint(*lengths), rng)
        out.append(Trace(objects, states, actions))
    return out


def random_actions(domain: Domain, objects, rng: random.Random, n: int):
    """n random grounded actions (distinct, type-compatible objects), as (name, args)."""
    from rosame import Instance
    acts = Instance(domain, objects).actions
    return [(a.action_schema.name, tuple(o.name for o in a.args)) for a in (rng.choice(acts) for _ in range(n))]


def naive_propositions(domain: Domain, objects) -> List[Tuple[str, Tuple[str, ...]]]:
    """Proposition order by definition: permutations filtered by type."""
    types = {n: domain.get_type(t) for n, t in objects}
    names = [n for n, _ in objects]
    return [(p.name, args) for p in domain.predicates for args in permutations(names, p.arity)
            if all(types[o].is_child_of(t) for o, t in zip(args, p.types))]


def naive_ground(model, domain: Domain, objects, action) -> torch.Tensor:
    """[3, P] grounded (pre, add, del) of one action call, slot by slot, from the role probabilities."""
    probs = model.role_probabilities()
    props = {a: i for i, a in enumerate(naive_propositions(domain, objects))}
    out = torch.zeros(3, len(props))
    name, args = action
    for k, (a, p, x) in enumerate(model.triples):
        if a.name == name:
            j = props[(p.name, tuple(args[i - 1] for i in x))]
            out[:, j] += torch.stack([probs[k, 2] + probs[k, 3], probs[k, 1], probs[k, 3]])
    return out


def same_model(a: ActionModel, b: ActionModel) -> bool:
    return a == b and not a.diff(b)
