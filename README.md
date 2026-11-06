# ROSAME

**ROSAME** learns lifted STRIPS action models (the preconditions and effects of every action schema)
from traces of states and actions. States may be **probabilistic** (each proposition true with some
probability, for example the output of a classifier or a noisy sensor) as well as symbolic, and the
model is a PyTorch module, so it can be trained jointly with a perception network end to end.

- Learns from traces whose **object sets differ** and whose **lengths differ**, batched on CPU or GPU.
- Robust to noise: the learned model is a probability over roles for every lifted literal.
- Reads PDDL domains and problems, and writes learned models as PDDL or JSON.
- Depends only on PyTorch.

ROSAME was introduced in *Neuro-Symbolic Learning of Lifted Action Models from Visual Traces*
(Xi, Gould and Thiébaux, ICAPS 2024).

## Installation

```bash
pip install rosame
```

ROSAME needs PyTorch 2.0 or later. To use a GPU, install the PyTorch build for your platform first,
following [pytorch.org](https://pytorch.org/get-started/locally/).

The planning structures (`Domain`, `Instance`, `ActionModel`, `Trace`, ...) come from
[`planning-structs`](https://github.com/xikaioliver/planning-structs), which is installed with ROSAME
and imported as `planning_structs`.

## Quickstart

```python
import rosame
from planning_structs import Trace

# 1. The domain: types, predicates and action signatures. Preconditions and effects are what we learn,
#    so the PDDL may leave them empty (any present are ignored for learning).
pddl = rosame.io.read_pddl_domain("""
(define (domain switches)
  (:requirements :strips :typing)
  (:types switch - device room device)
  (:predicates (powered) (robot-at ?r - room) (in ?d - device ?r - room) (connected ?a ?b - room)
               (on ?s - switch) (off ?s - switch))
  (:action move     :parameters (?from ?to - room)       :precondition (and) :effect (and))
  (:action turn-on  :parameters (?s - switch ?r - room)  :precondition (and) :effect (and))
  (:action turn-off :parameters (?s - switch ?r - room)  :precondition (and) :effect (and)))
""")

# 2. Traces: objects, T + 1 states and T actions. Each trace may use different objects.
static = ["(powered)", "(in s1 r2)", "(connected r1 r2)", "(connected r2 r3)", "(connected r3 r1)"]
traces = [
    Trace(objects=[("r1", "room"), ("r2", "room"), ("r3", "room"), ("s1", "switch")],
          states=[static + ["(robot-at r1)", "(off s1)"],
                  static + ["(robot-at r2)", "(off s1)"],
                  static + ["(robot-at r2)", "(on s1)"],
                  static + ["(robot-at r2)", "(off s1)"]],
          actions=["(move r1 r2)", "(turn-on s1 r2)", "(turn-off s1 r2)"]),
    # ... more traces, with other rooms and switches
]

# 3. Learn, then inspect or save the model.
result = rosame.learn(pddl.domain, traces)
print(result.action_model)
result.model.to_pddl(pddl_domain=pddl, path="learned.pddl")    # keeps your parameter names
```

States can be given as sets of true atoms (above), as `{atom: probability}` dictionaries, or as
tensors of probabilities over the trace's propositions:

```python
Trace(objects, states=[{"(robot-at r1)": 0.94, "(on s1)": 0.12, "(off s1)": 0.85}, ...], actions=[...])
```

## Three ways to use ROSAME

The package is layered; use the level that fits your project.

**1. `rosame.learn`**: train on traces in one call (above). Options cover the loss
hyperparameters, early stopping, device and memory budget; see the [user guide](docs/user_guide.md).

**2. Your own training loop** with the building blocks, for example to train a perception model
jointly. The grounder turns a batch of traces with different object sets into index tensors; ROSAME
grounds the whole batch at once; `ROSAMELoss` is the objective of the paper.

```python
from rosame import Grounder, ROSAME, ROSAMELoss

grounder, model = Grounder(domain), ROSAME(domain)
loss_fn = ROSAMELoss(prior_weight=0.2, endpoint_weight=1.0, reduction="transition")
optimizer = torch.optim.Adam(list(model.parameters()) + list(perception.parameters()), lr=1e-3)

batch = grounder.collate(objects_per_trace, actions_per_trace)      # in your DataLoader's collate_fn
states = perception(images)                                          # [B, T + 1, P_max] probabilities
pre, add, dele = model(batch)                                        # [B, T, P_max], padding is 0
loss = loss_fn(pre, add, dele, states[:, :-1], states[:, 1:], batch)
loss.backward(); optimizer.step()
```

**3. The model alone.** `ROSAME(batch)` returns grounded precondition / add / delete probabilities for
every proposition of every transition. Use any objective you like.

## Data formats

| | Form |
|---|---|
| objects | `[("r1", "room"), ("s1", "switch"), ...]`, including any domain constants the states use |
| atom | `"(in s1 r2)"`, `"in s1 r2"`, `("in", "s1", "r2")` or `("in", ("s1", "r2"))` |
| state | iterable of true atoms, `{atom: probability}`, or a tensor `[P]` |
| action | an atom-like call such as `"(move r1 r2)"`, or `None` for an unknown action (that step is skipped) |
| proposition order | `Grounder(domain).instance(objects).propositions`, needed only for tensor states |

Traces can be stored as JSON lines (`rosame.io.read_traces_jsonl` / `write_traces_jsonl`), and
domains as PDDL or as a JSON specification (`rosame.io.read_domain_json`).

## What ROSAME assumes

STRIPS with typing: positive preconditions, add and delete effects, and every literal of an action
schema refers only to the action's parameters (no constants), each parameter bound to a distinct
object. A literal plays exactly one of four roles: none, add effect, precondition, or precondition
and delete effect. Transitions whose action repeats an object are skipped. The
[user guide](docs/user_guide.md) explains these assumptions and their consequences in detail.

## Documentation

- [User guide](docs/user_guide.md): concepts, data, batching, the loss and its hyperparameters,
  learning options, PDDL/JSON input and output, joint training with perception, troubleshooting.
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md).

## Code for the papers

The code used for each paper is kept unchanged on its own branch and tag:

| Paper | Branch | Tag |
|---|---|---|
| Neuro-Symbolic Learning of Lifted Action Models from Visual Traces (ICAPS 2024) | `paper/icaps24` | `icaps24` |
| Learning Lifted Action Models from Unsupervised Visual Traces (ICAPS 2026) | `paper/icaps26` | `icaps26` |

The `icaps24` tag marks the code exactly as published; the `paper/icaps24` branch also contains
later fixes.

## Citation

```bibtex
@inproceedings{xgt:24,
  author    = {Kai Xi and Stephen Gould and Sylvie Thi{\'{e}}baux},
  title     = {Neuro-Symbolic Learning of Lifted Action Models from Visual Traces},
  booktitle = {Proceedings of the 34th International Conference on Automated Planning and Scheduling ({ICAPS}-24)},
  year      = {2024},
  pages     = {653-662}
}

@inproceedings{xgt:26,
  author    = {Kai Xi and Stephen Gould and Sylvie Thi{\'{e}}baux},
  title     = {Learning Lifted Action Models from Unsupervised Visual Traces},
  booktitle = {Proceedings of the 36th International Conference on Automated Planning and Scheduling, ({ICAPS}-26)},
  year      = {2026}
}
```

## License

See [LICENSE](LICENSE).
