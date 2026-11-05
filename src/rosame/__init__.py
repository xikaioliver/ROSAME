"""
ROSAME: neuro-symbolic learning of lifted STRIPS action models from (probabilistic) state traces.

Layers, from lowest to highest (each usable on its own):
    rosame.planning_structs  Domain, Instance, ActionModel: typed STRIPS structures
    rosame.grounding         Grounder, GroundedBatch: batches of traces over different object sets
    rosame.model             ROSAME: the lifted probabilistic action model (a torch.nn.Module)
    rosame.losses            ROSAMELoss / rosame_loss: the objective of the ROSAME paper
    rosame.learn             learn(): train on traces in one call
    rosame.io                PDDL and JSON input/output
"""
from importlib.metadata import PackageNotFoundError, version

from rosame import io
from rosame.grounding import GroundedAction, GroundedBatch, Grounder
from rosame.learn import LearnResult, Trace, learn
from rosame.losses import ROSAMELoss, rosame_loss
from rosame.model import ROLES, ROSAME
from rosame.planning_structs import ActionModel, ActionSchema, Domain, Instance, Predicate, Type

try:
    __version__ = version("rosame")
except PackageNotFoundError:          # running from a source checkout without installation
    __version__ = "0.0.0+unknown"

__all__ = ["ROSAME", "ROLES", "Grounder", "GroundedBatch", "GroundedAction", "ROSAMELoss", "rosame_loss",
           "learn", "Trace", "LearnResult", "Domain", "Instance", "ActionModel", "ActionSchema", "Predicate",
           "Type", "io", "__version__"]
