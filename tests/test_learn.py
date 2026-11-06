import random
import warnings

import pytest
import torch
from helpers import reference_action_model, same_model, switches_domain, switches_traces

import rosame
from planning_structs import Trace
from rosame import Grounder, ROSAMELoss, learn

pytestmark = pytest.mark.slow
FAST = dict(epochs=800, min_epochs=100, patience=100)


def test_learns_from_traces_with_different_object_sets_and_lengths():
    d = switches_domain()                   # includes prevail conditions, which need the prior
    result = learn(d, switches_traces(random.Random(0), 60), **FAST)
    assert same_model(result.action_model, reference_action_model(d)), result.action_model.diff(reference_action_model(d))
    assert result.stats["used_transitions"] == result.stats["transitions"] and result.stats["stopped_early"]
    first, last = result.history[0], result.history[-1]
    assert last["prediction"] < first["prediction"] / 10 and last["applicability"] < first["applicability"] / 5


def test_learns_from_noisy_probabilistic_states():
    d = switches_domain()
    g = Grounder(d)
    rng = torch.Generator().manual_seed(0)
    noisy = []
    for tr in switches_traces(random.Random(1), 60):
        clean = torch.stack([g.encode_state(tr.objects, s) for s in tr.states])
        noise = torch.rand(clean.shape, generator=rng) * 0.2          # true atoms in [0.8, 1], false in [0, 0.2]
        noisy.append(Trace(tr.objects, (clean - noise).abs(), tr.actions))
    result = learn(d, noisy, **FAST)
    assert same_model(result.action_model, reference_action_model(d)), result.action_model.diff(reference_action_model(d))


def test_trace_reduction_also_learns():
    d = switches_domain()
    result = learn(d, switches_traces(random.Random(2), 60), loss=ROSAMELoss(reduction="trace"), **FAST)
    assert same_model(result.action_model, reference_action_model(d)), result.action_model.diff(reference_action_model(d))


def test_stats_and_warnings_for_unusable_data():
    d = switches_domain()
    objs = [("r0", "room"), ("r1", "room")]
    traces = [Trace(objs, [["(robot-at r0)", "(connected r0 r1)"], ["(robot-at r1)", "(connected r0 r1)"],
                           ["(robot-at r1)", "(connected r0 r1)"]], ["(move r0 r1)", "(move r1 r1)"])]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = learn(d, traces, epochs=5, min_epochs=1, patience=1)
    text = " ".join(str(x.message) for x in w)
    assert "repeats an object" in text and "turn-on" in text
    assert result.stats["ignored_repeated_objects"] == 1 and result.stats["transitions_per_schema"]["move"] == 1
    with pytest.raises(ValueError):
        learn(d, [Trace(objs, [["(robot-at r0)"]], ["(move r0 r1)"])])            # T + 1 states needed


def test_learn_from_jsonl_file(tmp_path):
    d = switches_domain()
    rosame.io.write_traces_jsonl(switches_traces(random.Random(3), 10), tmp_path / "t.jsonl")
    result = learn(d, rosame.io.read_traces_jsonl(tmp_path / "t.jsonl"), epochs=50, min_epochs=10, patience=10)
    assert "(:action turn-on" in result.to_pddl()
