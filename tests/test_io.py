import pytest
import torch
from helpers import SWITCHES_PDDL, reference_action_model, switches_domain

from planning_structs import Trace
from rosame import Grounder
from rosame.io import (read_action_model_json, read_domain_json, read_objects_json, read_pddl_domain, read_pddl_problem,
                       read_traces_jsonl, write_action_model_json, write_domain_json, write_objects_json, write_pddl,
                       write_traces_jsonl)

WITH_CONSTANTS = """; a comment before the definition
(define (domain hub)
  (:requirements :strips :typing :negative-preconditions)
  (:types cart - vehicle vehicle place)
  (:constants depot - place)
  (:predicates (at ?v - vehicle ?p - place) (loaded ?c - cart))
  (:action load :parameters (?c - cart)
    :precondition (and (at ?c depot) (not (loaded ?c)))
    :effect (loaded ?c)))
"""


def test_read_domain_and_its_action_model():
    pd = read_pddl_domain(SWITCHES_PDDL)
    d = pd.domain
    assert d.name == "switches" and d.get_type("switch").is_child_of(d.get_type("device"))
    assert pd.action_parameters["turn-on"] == ["?s", "?r"]
    assert pd.action_model == reference_action_model(d) and pd.unsupported == []


def test_types_constants_and_unsupported_literals():
    pd = read_pddl_domain(WITH_CONSTANTS)
    assert pd.domain.get_type("cart").is_child_of(pd.domain.get_type("vehicle")) and pd.constants == [("depot", "place")]
    assert pd.action_model.add_effects["load"][0][0].name == "loaded"
    assert len(pd.unsupported) == 2                     # a constant in a precondition; a negative precondition
    prob = read_pddl_problem("(define (problem p) (:domain hub) (:objects c1 - cart) "
                             "(:init (at c1 depot)) (:goal (and (loaded c1))))", pd)
    assert prob.objects == [("c1", "cart"), ("depot", "place")]
    assert prob.init == [("at", ("c1", "depot"))] and prob.goal == [("loaded", ("c1",))]


def test_pddl_round_trip_with_and_without_the_original_header(tmp_path):
    pd = read_pddl_domain(SWITCHES_PDDL)
    path = tmp_path / "learned.pddl"
    text = write_pddl(pd.action_model, path=path, pddl_domain=pd)
    back = read_pddl_domain(path)
    assert path.read_text() == text and back.action_parameters == pd.action_parameters
    assert back.action_model == pd.action_model
    plain = read_pddl_domain(write_pddl(pd.action_model))
    assert plain.action_parameters["turn-on"] == ["?a", "?b"] and plain.action_model == pd.action_model


def test_json_round_trips(tmp_path):
    d = switches_domain()
    write_domain_json(d, tmp_path / "d.json")
    d2 = read_domain_json(tmp_path / "d.json")
    assert [(a.name, [t.name for t in a.types]) for a in d2.action_schemas] == \
           [(a.name, [t.name for t in a.types]) for a in d.action_schemas]
    assert d2.get_type("switch").is_child_of(d2.get_type("device"))
    objs = [("r0", "room"), ("s0", "switch")]
    write_objects_json(objs, tmp_path / "o.json")
    assert read_objects_json(tmp_path / "o.json") == objs
    am = reference_action_model(d)
    write_action_model_json(am, tmp_path / "m.json")
    assert read_action_model_json(tmp_path / "m.json", d) == am


def test_traces_jsonl_round_trip(tmp_path):
    d = switches_domain()
    objs = [("r0", "room"), ("r1", "room"), ("s0", "switch")]
    symbolic = Trace(objs, [["(robot-at r0)", "(connected r0 r1)"], {"(robot-at r1)": 0.9}], ["(move r0 r1)"])
    tensor_states = torch.zeros(2, Grounder(d).propositions(objs).num_props)
    tensor_states[0, 0], tensor_states[1, 3] = 1.0, 0.5
    probabilistic = Trace(objs, tensor_states, [None])
    write_traces_jsonl([symbolic, probabilistic], tmp_path / "t.jsonl", domain=d)
    back = read_traces_jsonl(tmp_path / "t.jsonl")
    assert back[0].states == [["(robot-at r0)", "(connected r0 r1)"], {"(robot-at r1)": 0.9}]
    assert back[0].actions == ["(move r0 r1)"]
    g = Grounder(d)
    assert torch.allclose(torch.stack([g.encode_state(objs, s) for s in back[1].states]), tensor_states)
    with pytest.raises(ValueError):
        write_traces_jsonl([probabilistic], tmp_path / "x.jsonl")


def test_action_model_equality_and_diff():
    pd = read_pddl_domain(SWITCHES_PDDL)
    am = reference_action_model(pd.domain)
    assert am == pd.action_model and am.diff(pd.action_model) == []
    am.add_effects["move"] = am.add_effects["move"] + [(pd.domain.get_predicate("powered"), ())]
    assert am != pd.action_model
    assert am.diff(pd.action_model) == ["move: add effect (powered) only in this model"]
