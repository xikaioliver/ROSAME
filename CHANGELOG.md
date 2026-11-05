# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
