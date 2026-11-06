# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-11-06

Data in its own order, and planning objects as inputs. Everything is optional: existing code and
learned results are unchanged.

### Added
- Traces may declare the order of their data (`planning_structs.Trace.proposition_order` and
  `action_order`, from `planning-structs` 0.2.0): vector states are reordered into the package's
  proposition order, and integer actions are looked up in the trace's own action list. Duplicates
  are errors; propositions the order does not name are errors unless
  `learn(..., missing_propositions="zero")`; atoms without a proposition are dropped with a warning.
- `Grounder.reorderer(objects, proposition_order)`: a differentiable reordering (and its inverse) of
  the last dimension of tensors, e.g. for a perception network that keeps its own output order.
- `Grounder.action_index_map(objects, action_order)`: maps a data's own action indices to
  `Instance.actions` indices (-1 for actions that repeat an object).
- `Grounder.collate(..., action_orders=...)` and `Grounder.ground_index`.
- `planning_structs.Action` and `Proposition` objects are accepted wherever actions and atoms are.
- `learn` accepts integer actions (indices into `action_order`, or into `Instance.actions`).
- JSON-lines traces may store vector states with `"proposition_order"` and integer actions with
  `"action_order"`.

### Changed
- Requires `planning-structs>=0.2.0,<0.3`.

## [0.2.0] - 2026-11-06

The planning structures now live in their own package,
[`planning-structs`](https://github.com/xikaioliver/planning-structs) (imported as `planning_structs`),
which ROSAME depends on and installs automatically.

### Changed
- **Breaking:** `Domain`, `Type`, `Predicate`, `ActionSchema`, `Instance`, `Object`, `Proposition`,
  `Action`, `ActionModel` and `Trace` are imported from `planning_structs` instead of `rosame`. Code
  written for 0.1.0 can keep requiring `rosame==0.1.0`.

### Removed
- The module `rosame.planning_structs` and the top-level names `rosame.Domain`, `rosame.Instance`,
  `rosame.ActionModel`, `rosame.ActionSchema`, `rosame.Predicate`, `rosame.Type` and `rosame.Trace`.

### Migration
- `from rosame import Domain, Trace` → `from planning_structs import Domain, Trace`
- `from rosame.planning_structs import Instance` → `from planning_structs import Instance`

The model, grounding, losses, `learn`, `rosame.io` and saved checkpoints are unchanged.

## [0.1.0] - 2026-11-05

First release as a Python package.

### Added
- `rosame.ROSAME`: lifted probabilistic action model as a PyTorch module, grounding whole batches of
  traces at once, including traces with different object sets and different lengths.
- `rosame.Grounder` and `rosame.GroundedBatch`: grounding of observed actions only (no enumeration of
  all groundings), proposition tables cached per object-set signature, and masks for padding.
- `rosame.ROSAMELoss` / `rosame.rosame_loss`: the ICAPS 2024 objective for padded batches, with
  `prior_weight`, `endpoint_weight` and the `"transition"` / `"trace"` reductions.
- `rosame.learn`: training on symbolic or probabilistic traces in one call.
- `rosame.planning_structs`: typed STRIPS domains, instances and discrete action models.
- `rosame.io`: PDDL domains, problems and models; JSON domain specifications, objects, models and
  JSON-lines traces.

### Changed (compared with the code of the ICAPS 2024 paper)
- Domain specifications list parameter types in order (the earlier format grouped them by type).
- Grounded actions of an `Instance` are built on first use.
