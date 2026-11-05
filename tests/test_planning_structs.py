import random
from itertools import permutations

from helpers import fixture_domain, fixture_objects, naive_propositions, switches_domain

from rosame import Instance


def test_proposition_order_is_permutations_filtered_by_type():
    rng = random.Random(0)
    for d, objs in ((fixture_domain(), fixture_objects(rng)),
                    (switches_domain(), [("r0", "room"), ("s0", "switch"), ("r1", "room"), ("s1", "switch")])):
        inst = Instance(d, objs)
        assert [(p.predicate.name, tuple(o.name for o in p.args)) for p in inst.propositions] == naive_propositions(d, objs)
        assert all(inst.prop_index[(p.predicate, p.args)] == i for i, p in enumerate(inst.propositions))


def test_actions_are_lazy_and_in_permutation_order():
    d = fixture_domain()
    objs = fixture_objects(random.Random(1))
    inst = Instance(d, objs)
    assert inst._actions is None                       # nothing enumerated yet
    types = {n: d.get_type(t) for n, t in objs}
    names = [n for n, _ in objs]
    expected = [(a.name, args) for a in d.action_schemas for args in permutations(names, a.arity)
                if all(types[o].is_child_of(t) for o, t in zip(args, a.types))]
    assert [(a.action_schema.name, tuple(o.name for o in a.args)) for a in inst.actions] == expected


def test_bindings_are_distinct_and_type_compatible():
    d = fixture_domain()
    for (a, p), bindings in d.predicate_arguments.items():
        for x in bindings:
            assert len(set(x)) == len(x) == p.arity
            assert all(a.types[i - 1].is_child_of(t) for i, t in zip(x, p.types))
    z, p2 = d.get_action_schema("z"), d.get_predicate("p2")
    assert d.predicate_arguments[(z, p2)] == [(1, 3), (2, 3)]      # B is an A
    assert d.predicate_arguments[(d.get_action_schema("y"), d.get_predicate("p3"))] == []   # needs B, B, C
