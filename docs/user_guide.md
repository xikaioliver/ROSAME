# ROSAME user guide

1. [Concepts](#1-concepts)
2. [Defining a domain](#2-defining-a-domain)
3. [Traces](#3-traces)
4. [Learning with `rosame.learn`](#4-learning-with-rosamelearn)
5. [The loss and its hyperparameters](#5-the-loss-and-its-hyperparameters)
6. [Batches and your own training loop](#6-batches-and-your-own-training-loop)
7. [Joint training with a perception model](#7-joint-training-with-a-perception-model)
8. [Reading the learned model](#8-reading-the-learned-model)
9. [Assumptions and limitations](#9-assumptions-and-limitations)
10. [Performance](#10-performance)
11. [Troubleshooting](#11-troubleshooting)

## 1. Concepts

**Domain.** Types (single inheritance, everything under `object`), predicates and action schemas,
each with typed parameters. ROSAME learns the preconditions and effects; it only needs the
signatures.

**Lifted slots.** For every action schema `a` and predicate `p`, ROSAME considers every way to fill
`p`'s positions with distinct, type-compatible parameters of `a`. Each such *binding* is one slot,
written as a tuple of 1-based parameter positions: in `move(?from, ?to)`, the literal
`(connected ?to ?from)` is the slot `(connected, (2, 1))`. `model.triples` lists all slots as `(schema, predicate, binding)`.

**Roles.** For every slot, ROSAME learns a probability over four roles:

| role | meaning |
|---|---|
| `none` | the literal is irrelevant to the action |
| `add` | add effect |
| `pre` | precondition, unchanged by the action |
| `pre+del` | precondition and delete effect |

so `P(pre) = P(pre) + P(pre+del)`, `P(add) = P(add)` and `P(del) = P(pre+del)`. Each schema has a
small neural network (PAM) that outputs these four probabilities for each of its slots.

**Object sets and propositions.** A trace's objects determine its propositions: every predicate
applied to distinct, type-compatible objects. Their order is fixed: predicates in domain order,
and for each, object tuples in lexicographic order of the objects' positions in your object list.
Only the sequence of object *types* matters, not the names, so traces whose object lists have the
same types in the same order (say, three rooms followed by two switches) share the same proposition
layout. `Grounder(domain).instance(objects).propositions` lists them.

**Grounding.** Grounding the action `turn-on(s1, r2)` copies the slot probabilities of `turn-on` to
the propositions they refer to in that trace, `(in s1 r2)`, `(robot-at r2)` and so on; every other proposition
gets 0. The *values* depend only on the schema and the *positions* only on the object set, which is
what lets ROSAME process traces over different object sets in one batch.

## 2. Defining a domain

From PDDL (preconditions and effects may be empty or absent; they are not used for learning):

```python
pddl = rosame.io.read_pddl_domain("domain.pddl")       # or the PDDL text itself
domain = pddl.domain
```

`pddl` also keeps the requirements, types, constants and parameter names, so a learned model can be
written back in the same shape (section 8). If the file does contain preconditions and effects,
`pddl.action_model` holds them as a `planning_structs.ActionModel`, which is handy for comparing a learned
model with a reference.

From a JSON specification:

```json
{"name": "switches",
 "types": [{"name": "room", "parent": null}, {"name": "device", "parent": null},
           {"name": "switch", "parent": "device"}],
 "predicates": [{"name": "robot-at", "types": ["room"]}, {"name": "in", "types": ["device", "room"]},
                {"name": "on", "types": ["switch"]}],
 "action_schemas": [{"name": "move", "types": ["room", "room"]}, {"name": "turn-on", "types": ["switch", "room"]}]}
```

```python
domain = rosame.io.read_domain_json("domain.json")
```

Or in Python:

```python
from planning_structs import Domain

domain = Domain(types=[("room", None), ("device", None), ("switch", "device")],
                predicates=[("robot-at", ["room"]), ("in", ["device", "room"]), ("on", ["switch"])],
                action_schemas=[("move", ["room", "room"]), ("turn-on", ["switch", "room"])],
                name="switches")
```

## 3. Traces

```python
from planning_structs import Trace

Trace(objects, states, actions)
```

**objects**: `[(name, type), ...]`. Include every object that states or actions mention, including
domain constants (`rosame.io.read_pddl_problem(path, pddl)` appends them for you).

**states**: `T + 1` states, in one of these forms:

- a list of states, each an iterable of true atoms: `["(robot-at r1)", "(in s1 r2)"]`;
- a list of states, each a dictionary of probabilities: `{"(robot-at r1)": 0.93, "(on s1)": 0.81}`;
  atoms not listed have probability 0;
- vectors of probabilities, as a tensor or array `[T + 1, P]` or a list of vectors, in the package's
  proposition order (section 1) or in the trace's own order (below).

Atoms may be written `"(in s1 r2)"`, `"in s1 r2"`, `("in", "s1", "r2")` or `("in", ("s1", "r2"))`. Atoms that
have no proposition (an unknown predicate or object, a type mismatch, or a repeated object such as
`(connected r1 r1)`) are ignored with a warning; `learn(..., on_unknown_atom="raise")` makes them errors.

**actions**: `T` action calls in the same forms as atoms, such as `"(move r1 r2)"`, or
`planning_structs.Action` objects, or integers (below). `None` marks an unknown action; that
transition is skipped. Calls that repeat an object (`(move r1 r1)`) are also
skipped with a warning, because ROSAME binds distinct objects to distinct parameters (section 9).

Atoms may also be `planning_structs.Proposition` objects. Objects inside `Action` and `Proposition`
objects are matched by name to the trace's objects.

**Data in its own order.** Vector states and integer actions normally follow the package's order:
`Grounder(domain).instance(objects).propositions` and `.actions`. Data that comes in its own order
(from a simulator, a dataset or another code base) declares it instead, and ROSAME reorders it:

```python
Trace(objects, states=vectors, actions=indices,
      proposition_order=["(robot-at r1)", "(robot-at r2)", ...],   # names of the vector entries
      action_order=["(move r1 r2)", "(move r2 r1)", ...])         # what each integer refers to
```

Duplicates in either order are errors. Propositions of the object set that `proposition_order`
does not name are errors too, unless `learn(..., missing_propositions="zero")` fills them with 0.
Atoms in the order with no proposition (e.g. repeated objects) are dropped with a warning
(`on_unknown_atom`). Generate order lists from the code that produced the data, never by hand.

Traces may have different object sets and different lengths. To store traces, use JSON lines:

```python
rosame.io.write_traces_jsonl(traces, "traces.jsonl", domain=domain)   # domain needed for tensor states
traces = rosame.io.read_traces_jsonl("traces.jsonl")
```

Each line looks like
`{"objects": [["r1", "room"], ["r2", "room"]], "states": [["(robot-at r1)"], {"(robot-at r2)": 0.9}], "actions": ["(move r1 r2)"]}`.
Traces with declared orders also store `"proposition_order"` (with states as vectors) and
`"action_order"` (with integer actions).

## 4. Learning with `rosame.learn`

```python
result = rosame.learn(domain, traces)
```

| argument | default | meaning |
|---|---|---|
| `loss` | `ROSAMELoss()` | the objective and its hyperparameters (section 5) |
| `epochs` | 2000 | maximum number of optimisation steps; every step uses all traces |
| `min_epochs`, `patience` | 200, 200 | stop once the most likely role of every slot has been unchanged for `patience` steps, after at least `min_epochs` |
| `lr` | 1e-3 | Adam learning rate |
| `seed` | 0 | seed for the model initialisation; `None` leaves the random number generator alone |
| `device` | `"cpu"` | for example `"cuda"` |
| `max_elements` | 4,000,000 | memory budget per chunk, in (state × proposition) entries; larger data is processed in chunks with gradient accumulation, which leaves the full-batch gradient unchanged |
| `model` | `None` | continue training an existing `ROSAME` |
| `model_kwargs` | `None` | network sizes for a new `ROSAME` (`pam_dim`, `pam_hidden`, `pam_depth`, `pam_dropout`, `trainable_inputs`, `use_layernorm`) |
| `on_unknown_atom` | `"warn"` | `"warn"`, `"ignore"` or `"raise"` |
| `missing_propositions` | `"error"` | for vector states with a `proposition_order` that misses propositions: `"error"` or `"zero"` |
| `verbose`, `log_every` | `False`, 100 | print and record the loss terms every `log_every` steps |

`result` is a `LearnResult`:

- `result.model`: the trained `ROSAME`;
- `result.action_model`: the discrete model (most likely role of every slot), a `planning_structs.ActionModel`;
- `result.to_pddl(...)`: the model as PDDL (section 8);
- `result.history`: the loss and its terms over training;
- `result.stats`: data coverage: transitions used and skipped, transitions per action schema,
  schemas without any usable transition, proposition counts, epochs, time.

Check `result.stats["schemas_without_data"]`: a schema that never occurs in usable transitions keeps
whatever its network was initialised to, and `learn` warns about it.

## 5. The loss and its hyperparameters

`rosame.ROSAMELoss` (a module) and `rosame.rosame_loss` (a function) implement the objective of the
ROSAME paper. For a transition `s1 --a--> s2` and the grounded `pre`, `add`, `del` of `a`, per
proposition:

```
prediction     (s1 * (1 - del) + (1 - s1) * add  -  s2)^2       the action turns s1 into s2
applicability  ((1 - s1) * pre)^2                               its preconditions hold in s1
prior          prior_weight * (1 - pre)^2                       prefer preconditions
```

`s1` and `s2` are probabilities (or 0/1 values).

```python
loss_fn = rosame.ROSAMELoss(prior_weight=0.2, endpoint_weight=1.0, reduction="transition")
loss = loss_fn(pre, add, dele, s1, s2, batch)
terms = loss_fn(pre, add, dele, s1, s2, batch, return_terms=True)   # loss, prediction, applicability, prior
```

**`prior_weight`** (λ in the paper, default 0.2). Some facts are true before and after every
occurrence of an action, so the data cannot tell whether they are preconditions or irrelevant; the
prior resolves this in favour of preconditions, which keeps learned models conservative. It also sets
how much noise is tolerated: a fact remains a precondition while it is false in fewer than roughly a
`prior_weight` fraction of the action's occurrences. Lower it if your data is clean and extra
preconditions are a problem; raise it for noisier data. Values above 1 are not recommended. 0
disables the prior.

**`endpoint_weight`** (γ in the paper, default 1.0). Extra weight on the terms that involve each
trace's first state (applicability of its first transition) and last state (prediction of its last
transition). Use it when those two states are known more reliably than the intermediate ones, for
example when intermediate states are predicted by a perception model but the initial and goal states
are given.

**`reduction`** (default `"transition"`). How entries from different traces are combined:

- `"transition"`: every real (transition, proposition) entry counts equally; the per-proposition
  terms are summed and divided by the number of real transitions.
- `"trace"`: each trace is averaged over its own `T × P` entries, then traces are averaged. This is
  the paper's mean squared error, and equals computing the loss for each trace separately and
  averaging.

When all traces have the same objects and length, the two differ only by a constant factor and
learn the same model. When they differ, `"trace"` gives each transition a weight of `1 / (T × P)`
for its trace, so evidence from traces with many propositions is outweighed by the prior from small
ones and spurious preconditions survive. Use `"transition"` unless you need the paper's exact
normalisation.

**Why the loss does not reach zero.** With `prior_weight > 0`, the prior pulls the precondition
probability of add effects and irrelevant literals towards 1 while the data pulls it towards 0, so
the optimum is a compromise with a small positive loss. The most likely role is still correct. Use
`return_terms=True` to watch the prediction and applicability terms.

Padding never contributes: only real propositions and real transitions count (`batch.prop_mask`,
`batch.step_mask`). The prior is summed over the propositions the action touches; for the others
`pre` is the constant 0, so including them, as the paper's formula does, would add a constant without
changing any gradient.

## 6. Batches and your own training loop

The `Grounder` turns traces into a `GroundedBatch` of index tensors; `ROSAME` grounds the whole batch
in one step.

```python
grounder = rosame.Grounder(domain)
batch = grounder.collate(objects_per_trace, actions_per_trace)
pre, add, dele = model(batch)                  # each [B, T_max, P_max]
```

`actions_per_trace[b]` may contain action calls, `None` (skip the step), indices into
`grounder.instance(objects).actions`, or the results of `grounder.ground_action`. The batch holds:

| field | shape | meaning |
|---|---|---|
| `schema` | `[B, T]` | action schema of each transition |
| `props` | `[B, T, K]` | proposition index of each slot in the trace's own proposition order |
| `slot_mask` | `[B, T, K]` | slots the schema actually has |
| `prop_mask` | `[B, P]` | real propositions of each trace |
| `step_mask` | `[B, T]` | real transitions |

`batch.touched()` gives the `[B, T, P]` propositions touched by each transition's action, and
`batch.to(device)` moves it. Padded propositions and steps of the outputs are exactly 0. Every real
entry is defined, including propositions the action does not touch (their `pre`, `add` and `del` are
constant 0), so a loss over all propositions trains whatever produced the states.

States go in a tensor `[B, T_max + 1, P_max]`:

```python
from rosame.utils import stack_padded, assemble_states
states = stack_padded([grounder.encode_state(objs, s) for s in trace_states])       # one trace: [T + 1, P]
states = stack_padded(list_of_state_tensors)                                         # batch: [B, T + 1, P_max]
```

`assemble_states(init, intermediate, final, batch.num_steps)` builds that tensor when each trace's
final state must be placed after its own last step (traces of different lengths).

A training step:

```python
loss_fn = rosame.ROSAMELoss()
for batch, states in loader:
    pre, add, dele = model(batch)
    loss = loss_fn(pre, add, dele, states[:, :-1], states[:, 1:], batch)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

With a `DataLoader`, build the batch in the `collate_fn`; the `Grounder` holds no tensors on a
device and can be used in worker processes, and `GroundedBatch` supports `pin_memory=True`.
`rosame.utils.BucketBatchSampler(sizes, batch_size)` groups traces of similar proposition counts to
limit padding. When you accumulate gradients over several batches with `reduction="transition"`,
pass the total number of transitions as `num_transitions` so that every transition keeps the same
weight.

**Data in its own order, in your own loop.** For integer actions from your own action list, pass
`grounder.collate(objects_list, actions_list, action_orders=[order, ...])`, or convert once with
`grounder.action_index_map(objects, order)` (a LongTensor from your indices to `Instance.actions`
indices; -1 for actions that repeat an object). For tensors in your own proposition order, build
`reorder = grounder.reorderer(objects, proposition_order)` once per object set:
`reorder(x)` maps the last dimension `[..., P_data]` to `[..., P]` and is differentiable, and
`reorder.inverse(y)` maps back.

## 7. Joint training with a perception model

ROSAME's loss is differentiable in the states, so a perception network that maps observations
(images, say) to proposition probabilities can be trained jointly with the action model, as in the
ROSAME paper. The network must output, for each trace, a probability for every proposition of that
trace's object set, in the proposition order of section 1, or in its own order passed through
`grounder.reorderer(objects, its_order)` (section 6); pad to `P_max` and let the masks handle the
rest. If the first and last states of each trace are known, feed them as given states and use
`endpoint_weight` to emphasise them.

## 8. Reading the learned model

```python
model.role_probabilities()      # [N, 4] probabilities of ROLES for each slot in model.triples
model.learned_roles()           # [N] most likely role per slot
model.action_model()            # planning_structs.ActionModel built from the most likely roles
print(model.action_model())     # readable summary
```

Write it out:

```python
model.to_pddl(path="learned.pddl")                      # generic header, parameters ?a ?b ...
model.to_pddl(pddl_domain=pddl, path="learned.pddl")    # header and parameter names of your domain file
rosame.io.write_action_model_json(model, "learned.json")
```

`model.state_dict()` saves the network as for any PyTorch module; `ROSAME(domain)` followed by
`load_state_dict` restores it.

For a single object set, the older interface is still available:
`model.set_instance(instance)` and then `model(action_indices)` returns `[len(action_indices), P]`
tensors, with indices into `instance.actions`.

## 9. Assumptions and limitations

- **STRIPS with typing.** Positive preconditions, add effects and delete effects. No negative or
  disjunctive preconditions, conditional effects, quantifiers or numeric fluents.
- **One role per literal.** A literal cannot be both an add and a delete effect, and a delete effect
  is always a precondition.
- **Literals over parameters only.** A literal of an action schema can only mention the schema's
  parameters, not constants, so a precondition like `(connected ?r hub)`, with `hub` a constant,
  cannot be learned. Constants
  can still appear in states.
- **Distinct objects per parameter.** Groundings bind distinct objects to distinct parameters.
  Transitions whose action repeats an object are skipped, which can leave a schema without data if
  it is always used that way.
- **Unobserved schemas.** A schema with no usable transition keeps its initial, random roles;
  `learn` warns and lists it in `result.stats`.
- **Always-true facts.** A fact that holds before and after every observed occurrence of an action
  becomes a precondition (the prior's choice). With little data this can add unnecessary
  preconditions; more varied traces remove them.

## 10. Performance

Each step evaluates every schema's network once, then grounds all transitions with one gather and
one scatter, so its cost grows with the batch size times the largest proposition count, not with the
number of possible groundings. Only actions that occur are grounded. Proposition tables are cached
per object-set signature.

Padding costs memory when object-set sizes vary a lot, since the number of propositions grows
quickly with the number of objects. Group traces of similar sizes (`BucketBatchSampler`) or rely on
`learn`'s chunking (`max_elements`).

## 11. Troubleshooting

**"no usable transitions for action schema(s) ..."**: those schemas never occur, or only with
repeated objects. Their learned roles are meaningless; add traces where they occur with distinct
objects.

**"atom(s) have no proposition in this object set"**: a state mentions an object missing from
`objects` (often a domain constant), a predicate missing from the domain, a repeated object, or the
wrong type.

**Too many preconditions**: see section 9 on always-true facts. More varied traces help, as does a
smaller `prior_weight` when the data is clean.

**Different results across seeds**: usually a schema with little or no data; check
`result.stats["transitions_per_schema"]`.
