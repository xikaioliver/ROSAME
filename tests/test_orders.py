"""Data in its own proposition and action order, and Action / Proposition objects as inputs (0.3.0).
The key property: any declared order gives exactly the same result as the package's own order."""
import random
import warnings

import pytest
import torch
from helpers import switches_domain, switches_problem, switches_traces

from planning_structs import Instance, Trace
from rosame import Grounder
from rosame.io import read_traces_jsonl, write_traces_jsonl
from rosame.learn import prepare_traces


def names_of(inst, objects):
    """Package-order atom strings and action strings of an object set, with the trace's names."""
    canon = {o: objects[i][0] for i, o in enumerate(inst.objects)}
    props = [f"({p.predicate.name}{''.join(' ' + canon[o] for o in p.args)})" for p in inst.propositions]
    acts = [f"({a.action_schema.name}{''.join(' ' + canon[o] for o in a.args)})" for a in inst.actions]
    return props, acts


def setup(seed=0):
    d = switches_domain()
    g = Grounder(d)
    objects = switches_problem(random.Random(seed))[0]
    return d, g, objects, Instance(d, objects)


def test_action_and_proposition_objects_equal_their_names():
    d, g, objects, inst = setup()
    acts = random.Random(1).sample(inst.actions, 4)
    by_object = g.collate([objects], [acts])
    by_name = g.collate([objects], [[(a.action_schema.name, [o.name for o in a.args]) for a in acts]])
    for f in ("schema", "props", "slot_mask", "prop_mask", "step_mask"):
        assert torch.equal(getattr(by_object, f), getattr(by_name, f))
    props = inst.propositions[:5]
    assert torch.equal(g.encode_state(objects, props),
                       g.encode_state(objects, [(p.predicate.name, [o.name for o in p.args]) for p in props]))


def test_reorderer_round_trip_and_gradients():
    d, g, objects, inst = setup()
    props, _ = names_of(inst, objects)
    perm = torch.randperm(len(props), generator=torch.Generator().manual_seed(0))
    order = [props[i] for i in perm.tolist()]                       # the data's own order
    x = torch.rand(3, 4, len(props))                                # package order
    data = x[..., perm]                                             # the same values in the data's order
    reorder = g.reorderer(objects, order)
    assert torch.equal(reorder(data), x) and torch.equal(reorder.inverse(x), data)
    data = data.clone().requires_grad_()
    (reorder(data) * torch.arange(len(props))).sum().backward()
    assert torch.equal(data.grad[0, 0], perm.float())               # each entry gets its own weight back


def test_reorderer_validation():
    d, g, objects, inst = setup()
    props, _ = names_of(inst, objects)
    with pytest.raises(ValueError, match="twice"):
        g.reorderer(objects, props + [props[0]])
    with pytest.raises(ValueError, match="not in proposition_order"):
        g.reorderer(objects, props[1:])
    filled = g.reorderer(objects, props[1:], missing="zero")(torch.ones(len(props) - 1))
    assert filled[0] == 0 and filled[1:].eq(1).all()
    extra = props + ["(connected r0 r0)"]                            # repeated objects: no proposition
    with pytest.warns(UserWarning, match="dropped"):
        r = g.reorderer(objects, extra)
    assert torch.equal(r(torch.arange(len(extra)).float()), torch.arange(len(props)).float())
    with pytest.raises(ValueError):
        g.reorderer(objects, extra, on_unknown="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g.reorderer(objects, extra, on_unknown="ignore")


def test_action_index_map_and_collate_with_action_orders():
    d, g, objects, inst = setup()
    _, acts = names_of(inst, objects)
    rng = random.Random(2)
    order = acts[:]
    rng.shuffle(order)
    order.append("(move r0 r0)")                                    # repeats an object
    mapping = g.action_index_map(objects, order)
    assert mapping[-1] == -1 and [acts[i] for i in mapping[:-1].tolist()] == order[:-1]
    data_idx = [rng.randrange(len(order) - 1) for _ in range(6)] + [len(order) - 1]
    a = g.collate([objects], [data_idx], action_orders=[order])
    b = g.collate([objects], [[order[i] for i in data_idx[:-1]] + [None]])
    c = g.collate([objects], [mapping[torch.tensor(data_idx)].tolist()])        # through the index map
    for f in ("schema", "props", "slot_mask", "step_mask"):
        assert torch.equal(getattr(a, f), getattr(b, f)) and torch.equal(getattr(a, f), getattr(c, f))
    with pytest.raises(ValueError, match="twice"):
        g.action_index_map(objects, order[:2] + order[:1])


def test_learn_prepares_identically_with_declared_orders():
    d = switches_domain()
    g = Grounder(d)
    plain, declared = [], []
    for k, tr in enumerate(switches_traces(random.Random(3), 8)):
        inst = Instance(d, tr.objects)
        props, acts = names_of(inst, tr.objects)
        states = torch.stack([g.encode_state(tr.objects, s) for s in tr.states])
        perm = torch.randperm(len(props), generator=torch.Generator().manual_seed(k))
        order_p = [props[i] for i in perm.tolist()]
        order_a = acts[:]
        random.Random(k).shuffle(order_a)
        names = [f"({a}{''.join(' ' + o for o in args)})" for a, args in tr.actions]
        plain.append(Trace(tr.objects, states, names))
        declared.append(Trace(tr.objects, states[:, perm].tolist(), [order_a.index(n) for n in names],
                              proposition_order=order_p, action_order=order_a))
    (ep, sp), (ed, sd) = prepare_traces(g, plain), prepare_traces(g, declared)
    assert sp == sd
    for (o1, s1, c1), (o2, s2, c2) in zip(ep, ed):
        assert o1 == o2 and torch.equal(s1, s2) and c1 == c2


def test_learn_accepts_integer_actions_in_instance_order():
    d = switches_domain()
    g = Grounder(d)
    tr = switches_traces(random.Random(4), 1)[0]
    inst = Instance(d, tr.objects)
    _, acts = names_of(inst, tr.objects)
    names = [f"({a}{''.join(' ' + o for o in args)})" for a, args in tr.actions]
    (_, _, calls_idx), = prepare_traces(g, [Trace(tr.objects, tr.states, [acts.index(n) for n in names])])[0]
    (_, _, calls_name), = prepare_traces(g, [Trace(tr.objects, tr.states, names)])[0]
    assert calls_idx == calls_name


def test_jsonl_round_trip_with_orders(tmp_path):
    d, g, objects, inst = setup()
    props, acts = names_of(inst, objects)
    tr = Trace(objects, [[0.0] * len(props), [1.0] * len(props)], [1], proposition_order=props, action_order=acts)
    write_traces_jsonl([tr], tmp_path / "t.jsonl")
    back = read_traces_jsonl(tmp_path / "t.jsonl")[0]
    assert back.proposition_order == props and back.action_order == acts and back.actions == [1]
    assert back.states == tr.states
