import random

import torch
from helpers import fixture_domain, fixture_objects, naive_ground, random_actions, switches_domain, switches_problem

from planning_structs import Instance
from rosame import Grounder, ROLES, ROSAME, rosame_loss
from rosame.io import read_pddl_domain


def random_batch(d, rng, n=6):
    objs_list, acts_list = [], []
    for _ in range(n):
        objs = fixture_objects(rng) if d.name == "fixture" else switches_problem(rng)[0]
        objs_list.append(objs)
        acts_list.append(random_actions(d, objs, rng, rng.randint(1, 5)))
    return objs_list, acts_list


def test_batched_grounding_matches_naive_reference():
    for d in (fixture_domain(), switches_domain()):
        torch.manual_seed(0)
        model, g = ROSAME(d), Grounder(d)
        objs_list, acts_list = random_batch(d, random.Random(0))
        gb = g.collate(objs_list, acts_list)
        out = torch.stack(model(gb)).detach()                    # [3, B, T, P_max]
        for b, (objs, acts) in enumerate(zip(objs_list, acts_list)):
            P = g.propositions(objs).num_props
            for t, a in enumerate(acts):
                assert torch.allclose(out[:, b, t, :P], naive_ground(model, d, objs, a).detach(), atol=1e-7)
            assert out[:, b, :, P:].abs().sum() == 0              # padded propositions
            assert out[:, b, len(acts):].abs().sum() == 0         # padded steps


def test_single_instance_api_matches_batched():
    d = fixture_domain()
    torch.manual_seed(0)
    model, g = ROSAME(d), Grounder(d)
    objs = fixture_objects(random.Random(2))
    inst = Instance(d, objs)
    model.set_instance(inst)
    idx = list(range(0, inst.num_actions, 3))
    assert torch.equal(torch.stack(model(idx)), torch.stack(model(g.collate([objs], [idx])))[:, 0])


def test_gradients_reach_every_real_state_entry_and_no_padding():
    d = switches_domain()
    torch.manual_seed(0)
    model, g = ROSAME(d), Grounder(d)
    objs_list, acts_list = random_batch(d, random.Random(1))
    gb = g.collate(objs_list, acts_list)
    T, P = gb.schema.shape[1], gb.num_props_max
    states = torch.rand(len(objs_list), T + 1, P, requires_grad=True)
    pre, add, dele = model(gb)
    rosame_loss(pre, add, dele, states[:, :-1], states[:, 1:], gb).backward()
    for b, acts in enumerate(acts_list):
        Pb = int(gb.prop_mask[b].sum())
        assert (states.grad[b, :len(acts) + 1, :Pb] != 0).all()    # touched or not, every real entry
        assert states.grad[b, :, Pb:].abs().sum() == 0
        assert states.grad[b, len(acts) + 1:].abs().sum() == 0
    assert all(p.grad is not None for p in model.parameters())


def test_state_dict_round_trip():
    d = fixture_domain()
    torch.manual_seed(0)
    a = ROSAME(d)
    assert not any("schema_triples" in k for k in a.state_dict())   # derived buffer, not saved
    torch.manual_seed(1)
    b = ROSAME(d)
    b.load_state_dict(a.state_dict(), strict=True)
    assert torch.equal(a.lifted_effects(), b.lifted_effects())


def test_action_model_follows_most_likely_roles_and_round_trips_through_pddl():
    d = fixture_domain()
    torch.manual_seed(0)
    model = ROSAME(d)
    am, roles = model.action_model(), model.learned_roles().tolist()
    for (a, p, x), r in zip(model.triples, roles):
        pre, add, dele = am.schema(a.name)
        assert ((p, x) in pre) == (ROLES[r] in ("pre", "pre+del"))
        assert ((p, x) in add) == (ROLES[r] == "add")
        assert ((p, x) in dele) == (ROLES[r] == "pre+del")
    assert read_pddl_domain(model.to_pddl()).action_model == am
