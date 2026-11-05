import random

import pytest
import torch
from helpers import naive_ground, random_actions, switches_domain, switches_problem
from test_model import random_batch

from rosame import Grounder, ROSAME, ROSAMELoss, rosame_loss
from rosame.losses import step_weights


def setup(seed=0, n=6, equal=False):
    d = switches_domain()
    torch.manual_seed(seed)
    model, g = ROSAME(d), Grounder(d)
    rng = random.Random(seed)
    if equal:
        objs = switches_problem(rng)[0]
        objs_list = [objs] * n
        acts_list = [random_actions(d, objs, rng, 3) for _ in range(n)]
    else:
        objs_list, acts_list = random_batch(d, rng, n)
    gb = g.collate(objs_list, acts_list)
    states = [torch.rand(len(a) + 1, g.propositions(o).num_props, generator=torch.Generator().manual_seed(k)).requires_grad_()
              for k, (o, a) in enumerate(zip(objs_list, acts_list))]
    return d, model, g, gb, objs_list, acts_list, states


def padded(states, gb):
    out = torch.zeros(len(states), gb.schema.shape[1] + 1, gb.num_props_max)
    for b, s in enumerate(states):
        out[b, :s.shape[0], :s.shape[1]] = s
    return out


def naive_terms(model, d, objs, acts, s, prior_weight):
    """One trace, unpadded, transition by transition: summed terms and the T * P entry count."""
    total = 0.0
    for t, a in enumerate(acts):
        pre, add, dele = naive_ground(model, d, objs, a)
        touched = pre + add + dele > 0
        s1, s2 = s[t], s[t + 1]
        total = total + ((s1 * (1 - dele) + (1 - s1) * add - s2) ** 2).sum() + (((1 - s1) * pre) ** 2).sum() \
            + prior_weight * ((1 - pre) ** 2 * touched).sum()
    return total, len(acts) * s.shape[1]


def grads(model, states):
    return [p.grad.clone() for p in model.parameters() if p.grad is not None] + [s.grad.clone() for s in states]


def zero(model, states):
    model.zero_grad()
    for s in states:
        s.grad = None


@pytest.mark.parametrize("reduction", ["trace", "transition"])
def test_reductions_match_naive_reference(reduction):
    d, model, g, gb, objs_list, acts_list, states = setup()
    terms = [naive_terms(model, d, o, a, s, 0.2) for o, a, s in zip(objs_list, acts_list, states)]
    if reduction == "trace":
        ref = sum(t / c for t, c in terms) / len(terms)
    else:
        ref = sum(t for t, _ in terms) / sum(len(a) for a in acts_list)
    ref.backward()
    g_ref = grads(model, states)
    zero(model, states)
    S = padded(states, gb)
    pre, add, dele = model(gb)
    loss = rosame_loss(pre, add, dele, S[:, :-1], S[:, 1:], gb, reduction=reduction)
    loss.backward()
    assert loss.item() == pytest.approx(ref.item(), rel=1e-5)
    # states were copied into the padded tensor; compare parameter gradients only
    n_params = sum(1 for p in model.parameters() if p.grad is not None)
    for a, b in zip(g_ref[:n_params], grads(model, states)[:n_params]):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-8)


def test_transition_equals_trace_times_P_when_sizes_are_equal():
    d, model, g, gb, objs_list, acts_list, states = setup(equal=True)
    S = padded(states, gb)
    pre, add, dele = model(gb)
    tr = rosame_loss(pre, add, dele, S[:, :-1], S[:, 1:], gb, reduction="trace")
    tn = rosame_loss(pre, add, dele, S[:, :-1], S[:, 1:], gb, reduction="transition")
    assert tn.item() == pytest.approx(tr.item() * gb.num_props_max, rel=1e-5)


def test_padding_values_do_not_matter():
    d, model, g, gb, *_ = setup()
    S = torch.rand(gb.schema.shape[0], gb.schema.shape[1] + 1, gb.num_props_max)
    real_p = gb.prop_mask[:, None, :].expand_as(S)
    real_t = torch.zeros_like(S, dtype=torch.bool)
    for b, n in enumerate(gb.num_steps.tolist()):
        real_t[b, :n + 1] = True
    noisy = torch.where(real_p & real_t, S, torch.rand_like(S) * 5)       # garbage in padding
    out = []
    for X in (S, noisy):
        X = X.clone().requires_grad_()
        pre, add, dele = model(gb)
        loss = rosame_loss(pre, add, dele, X[:, :-1], X[:, 1:], gb)
        loss.backward()
        out.append((loss.item(), X.grad))
    assert out[0][0] == pytest.approx(out[1][0])
    assert torch.allclose(out[0][1], out[1][1])
    assert out[1][1][~(real_p & real_t)].abs().sum() == 0


def test_endpoint_weight_uses_each_traces_own_first_and_last_real_step():
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 0]], dtype=torch.bool)
    w = step_weights(mask, 10.0)
    assert w.tolist() == [[10, 10, 0, 0], [10, 0, 1, 10], [0, 10, 10, 0], [0, 0, 100, 0]]
    assert step_weights(mask, 10.0, first=False).tolist()[1] == [1, 0, 1, 10]
    assert torch.equal(step_weights(mask, 1.0), mask.float())


@pytest.mark.parametrize("reduction", ["trace", "transition"])
def test_prior_gradient_matches_the_papers_definition(reduction):
    """Summing the prior over touched propositions only changes the value by a constant: under both
    reductions its gradient equals that of the paper's prior over all real propositions."""
    d, model, g, gb, *_ = setup()
    S = torch.rand(gb.schema.shape[0], gb.schema.shape[1] + 1, gb.num_props_max)
    pre, add, dele = model(gb)
    ours = rosame_loss(pre, add, dele, S[:, :-1], S[:, 1:], gb, reduction=reduction, return_terms=True)["prior"]
    real = gb.step_mask[:, :, None] & gb.prop_mask[:, None, :]
    full = 0.2 * (1 - pre) ** 2 * real                                                # all real entries
    paper = (full.sum((1, 2)) / real.sum((1, 2))).mean() if reduction == "trace" else full.sum() / gb.num_transitions
    g1 = torch.autograd.grad(ours, list(model.parameters()), retain_graph=True, allow_unused=True)
    g2 = torch.autograd.grad(paper, list(model.parameters()), allow_unused=True)
    for a, b in zip(g1, g2):
        if a is not None:
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-9)
    assert paper.item() > ours.item()                   # differs by a constant only


def test_terms_hyperparameters_and_validation():
    d, model, g, gb, *_ = setup()
    S = torch.rand(gb.schema.shape[0], gb.schema.shape[1] + 1, gb.num_props_max)
    pre, add, dele = model(gb)
    t = ROSAMELoss()(pre, add, dele, S[:, :-1], S[:, 1:], gb, return_terms=True)
    assert set(t) == {"loss", "prediction", "applicability", "prior"}
    assert t["loss"].item() == pytest.approx((t["prediction"] + t["applicability"] + t["prior"]).item())
    assert ROSAMELoss(prior_weight=0)(pre, add, dele, S[:, :-1], S[:, 1:], gb, return_terms=True)["prior"].item() == 0
    assert "reduction='transition'" in repr(ROSAMELoss())
    for kw in ({"prior_weight": -1}, {"endpoint_weight": 0}, {"reduction": "mean"}):
        with pytest.raises(ValueError):
            ROSAMELoss(**kw)
