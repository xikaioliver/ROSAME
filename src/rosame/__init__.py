"""
ROSAME: neuro-symbolic learning of lifted STRIPS action models from (probabilistic) state traces.

The planning structures (Domain, Instance, ActionModel, Trace, ...) come from the `planning-structs`
package (`import planning_structs`), installed with ROSAME. ROSAME's own layers, from lowest to
highest (each usable on its own):
    rosame.grounding   Grounder, GroundedBatch: batches of traces over different object sets
    rosame.model       ROSAME: the lifted probabilistic action model (a torch.nn.Module)
    rosame.losses      ROSAMELoss / rosame_loss: the objective of the ROSAME paper
    rosame.learn       learn(): train on traces in one call
    rosame.io          PDDL and JSON input/output
"""
from importlib.metadata import PackageNotFoundError, version

from rosame import io
from rosame.grounding import GroundedAction, GroundedBatch, Grounder
from rosame.learn import LearnResult, learn
from rosame.losses import ROSAMELoss, rosame_loss
from rosame.model import ROLES, ROSAME

try:
    __version__ = version("rosame")
except PackageNotFoundError:          # running from a source checkout without installation
    __version__ = "0.0.0+unknown"

__all__ = ["ROSAME", "ROLES", "Grounder", "GroundedBatch", "GroundedAction", "ROSAMELoss", "rosame_loss",
           "learn", "LearnResult", "io", "__version__"]
