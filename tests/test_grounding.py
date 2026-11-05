import random
import warnings

import pytest
import torch
from helpers import fixture_domain, fixture_objects, naive_propositions, switches_domain

from rosame import GroundedAction, Grounder

ROOMS = [("r0", "room"), ("r1", "room"), ("r2", "room"), ("s0", "switch")]


def test_ground_action_matches_all_grounded_actions():
    d = fixture_domain()
    g = Grounder(d)
    objs = fixture_objects(random.Random(0))
    props = {a: i for i, a in enumerate(naive_propositions(d, objs))}
    inst = g.instance(objs)
    for act in inst.actions:
        name, args = act.action_schema.name, [objs[inst.objects.index(o)][0] for o in act.args]
        ga = g.ground_action(objs, (name, args))
        expected = [props[(p.name, tuple(args[i - 1] for i in x))] for p, x in g.slots[act.action_schema]]
        assert ga.schema == d.action_schemas.index(act.action_schema) and list(ga.props) == expected


def test_action_forms_are_equivalent_and_errors_are_clear():
    g = Grounder(switches_domain())
    forms = ["(move r0 r1)", "move r0 r1", ("move", "r0", "r1"), ("move", ("r0", "r1")), ["move", ["r0", "r1"]]]
    assert len({g.ground_action(ROOMS, f) for f in forms}) == 1
    assert g.ground_action(ROOMS, "(move r0 r0)") is None          # repeated object
    for bad in ["(jump r0)", "(move r0 zz)", "(move r0)", "(turn-on r0 r1)"]:
        with pytest.raises(ValueError):
            g.ground_action(ROOMS, bad)


def test_states_encode_and_decode():
    g = Grounder(switches_domain())
    P = g.propositions(ROOMS).num_props
    s = g.encode_state(ROOMS, ["(powered)", ("in", "s0", "r1"), "robot-at r2"])
    assert s.shape == (P,) and s.sum() == 3
    assert set(g.decode_state(ROOMS, s)) == {("powered", ()), ("in", ("s0", "r1")), ("robot-at", ("r2",))}
    p = g.encode_state(ROOMS, {"(on s0)": 0.7, "(connected r0 r1)": 0.2})
    assert sorted(p[p > 0].tolist()) == pytest.approx([0.2, 0.7])
    assert torch.equal(g.encode_state(ROOMS, s), s)
    with pytest.warns(UserWarning):
        g.encode_state(ROOMS, ["(connected r0 r0)"])
    with pytest.raises(ValueError):
        g.encode_state(ROOMS, ["(connected r0 r0)"], on_unknown="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g.encode_state(ROOMS, ["(connected r0 r0)"], on_unknown="ignore")


def test_collate_formats_masks_and_touched():
    g = Grounder(switches_domain())
    small, large = ROOMS, ROOMS + [("r3", "room"), ("s1", "switch")]
    idx = [g.action_index(small, "(move r0 r1)"), g.action_index(small, "(turn-on s0 r1)")]
    by_index = g.collate([small], [idx])
    by_call = g.collate([small], [["(move r0 r1)", "(turn-on s0 r1)"]])
    by_grounded = g.collate([small], [[g.ground_action(small, "(move r0 r1)"), g.ground_action(small, "(turn-on s0 r1)")]])
    for f in ("schema", "props", "slot_mask", "prop_mask", "step_mask"):
        assert torch.equal(getattr(by_index, f), getattr(by_call, f))
        assert torch.equal(getattr(by_call, f), getattr(by_grounded, f))
    gb = g.collate([small, large], [["(move r0 r1)"], ["(move r3 r0)", None, "(turn-off s1 r3)", -1, "(move r1 r1)"]])
    assert gb.step_mask.tolist() == [[True, False, False, False, False], [True, False, True, False, False]]
    assert gb.prop_mask.sum(1).tolist() == [g.propositions(small).num_props, g.propositions(large).num_props]
    assert gb.num_transitions == 3
    t = gb.touched()
    assert t[0, 0].sum() == 5 and t[1, 2].sum() == 5 and not t[0, 1:].any()   # move: 5 slots; turn-off: 5 slots
    assert isinstance(g.ground_action(small, "(move r0 r1)"), GroundedAction)


def test_tables_are_shared_across_object_names():
    g = Grounder(switches_domain())
    a = g.collate([ROOMS], [["(move r0 r1)"]])
    b = g.collate([[("x", "room"), ("y", "room"), ("z", "room"), ("w", "switch")]], [["(move x y)"]])
    assert torch.equal(a.props, b.props) and len(g._prop_tables) == 1
