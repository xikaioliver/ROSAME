# ROSAME: a neu**RO**-**S**ymbolic **A**ction **M**odel l**E**arner

## Paper
This repository holds codes for our paper in ICAPS 2024: [Neuro-Symbolic Learning of Lifted Action Models from Visual Traces](https://users.cecs.anu.edu.au/~thiebaux/papers/icaps24-rosame.pdf).

## Use ROSAME as a PyTorch module
We implement ROSAME as a PyTorch module that can be plugged into any other deep learning networks.

### Create a domain model
```python
# Create a model for the Gripper domain.
from models.rosame import *

# Define all the types in the domain.
base = Type("object", None)
room = Type("room", base)
ball = Type("ball", base)
gripper = Type("gripper", base)

# A Domain model requires a list of types, a list of predicates, a list of action schemas, and a device (optional).
domain_model = Domain_Model(
    [
        room,
        ball,
        gripper,
    ],
    [
        # A predicate requires a name and a signature (mapping types to numbers of arguments).
        Predicate("at-robby", {room: 1}),
        Predicate("at", {ball: 1, room: 1}),
        Predicate("free", {gripper: 1}),
        Predicate("carry", {ball: 1, gripper: 1}),
    ],
    [
        # An action schema requires a name and a signature (mapping types to numbers of arguments).
        Action_Schema("move", {room: 2}),
        Action_Schema("pick", {ball: 1, room: 1, gripper: 1}),
        Action_Schema("drop", {ball: 1, room: 1, gripper: 1}),
    ],
    device=torch.device("cuda"),
)

# Alternatively, a domain model can be created from a supplied JSON file.
# domain_model = load_model("models/domains/gripper/domain_model.json", device=torch.device("cuda"))
```

### Ground the domain model
```python
# A domain model must be grounded on a set of objects before use. 
# Note that domain models can be regrounded on different sets of objects. This will not affect the learned lifted model.
# Objects are supplied in the form of a dictionary mapping each type to a list of objects of the type.
objects = {
    room: ["rooma", "roomb"],
    ball: ["ball1", "ball2", "ball3", "ball4", "ball5", "ball6"],
    gripper: ["left", "right"],
}
domain_model.ground(objects)

# Alternatively, a domain model can be grounded from a supplied JSON file describing the objects.
# domain_model.ground_from_json("models/domains/gripper/objects.json")
```

### Use the domain model
```python
# Get a list of predicates
domain_model.predicates
# Get a list of action schemas
domain_model.action_schemas
# We maintain an order of propositions.
# Get a dictionary mapping each proposition name to its index in the order.
domain_model.propositions
# We maintain an order of actions.
# Get a dictionary mapping each action name to its index in the order.
domain_model.actions
```

### Infer with ROSAME
```python
# The build() function expect a list of action indices (can be found by querying domain_model.actions with the action names)
# It returns the (grounded) preconditions, add and delete effects of the actions.
precon, addeff, deleff = domain_model.build([0, 1, 2])
```

### Print the action model
```python
for schema in domain_model.action_schemas:
    schema.pretty_print()
    print("----------------------------")
```

## Create Visual Traces
We include all our data creation scripts. The scripts can be run in the data folder.

**Blocks World (grid world)**  
This script requires [Blocks World Generator and Planner](https://gitlab.com/thiebaux/blocks-world-generator-and-planner) (Slaney and Thiebaux 2001) as a prerequisite.  
`python generate_blocksworld.py -o 5 -t 800 -l 10 -s data`

**Gripper (grid world)**  
This script requires [MACQ](https://github.com/AI-Planning/macq) (Callanan et al. 2022), which can be installed with `pip install macq`.  
`python generate_gripper.py -b 6 -t 1000 -l 5 --skip 1 -s data`

**Logistics (grid world)**  
This script also requires macq.  
`python generate_logistics.py --num_packages 6 -t 3000 -l 10 --skip 1 -s data`

**Blocks World (synthesized)**  
All scripts below that generate syntehsized visual traces are based on [PDDLGym](https://github.com/tomsilver/pddlgym) (Silver and Chitnis 2020).  
`python generate_blocksworld_pddlgym.py -s data`

**Towers of Hanoi (synthesized)**  
`python generate_hanoi_pddlgym.py -s data`

**Slide/8-puzzle (synthesized)**  
`python generate_slide_pddlgym.py -s data`

## Training ROSAME-I on Visual Traces
**Blocks World (grid world)**    
`python train.py --domain grid_blocks --gamma 10 --lambda_ 0.2 --epochs 200 --dataset_pth data/data --trace_num 800 --trace_len 10`  
**Gripper (grid world)**  
`python train.py --domain grid_gripper --gamma 10 --lambda_ 0.2 --epochs 100 --dataset_pth data/data --trace_num 1000 --trace_len 5`  
**Logistics (grid world)**  
`python train.py --domain grid_logistics --gamma 10 --lambda_ 0.2 --epochs 150 --dataset_pth data/data --trace_num 2500 --trace_len 10`  
**Blocks World (synthesized)**  
`python train.py --domain synth_block --gamma 10 --lambda_ 0.2 --epochs 100 --dataset_pth data/data --trace_num 100 --trace_len 10`  
**Towers of Hanoi (synthesized)**  
`python train.py --domain synth_hanoi --gamma 10 --lambda_ 0.2 --epochs 70 --dataset_pth data/data --trace_num 70 --trace_len 5`  
**Slide/8-puzzle (synthesized)**  
`python train.py --domain synth_8-puzzle --gamma 10 --lambda_ 0.4 --epochs 300 --dataset_pth data/data --trace_num 300 --trace_len 5` 

## Citation
Please use this bibtex if you want to cite this repository in your publications:
```
@inproceedings{silver2020pddlgym,
  author    = {Kai Xi and Stephen Gould and Sylvie Thi{\'{e}}baux},
  title     = {Neuro-Symbolic Learning of Lifted Action Models from Visual Traces},
  booktitle = {Proceedings of the Thirty-Fourth International Conference on Automated Planning and Scheduling, {ICAPS} 2024, Banff, Alberta, Canada, June 1-6, 2024},
  year      = {2024},
  url       = {https://gitlab.com/xikaioliver2/ROSAME},
}
```