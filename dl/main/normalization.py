import numpy as np
import numpy.random as random
from ..util.tuning import parameters
# from ..data.objutil import *


def normalize(x,save=True):
    if ("mean" in parameters) and save:
        mean = np.array(parameters["mean"][0])
        std  = np.array(parameters["std"][0])
    else:
        mean               = np.mean(x,axis=0)
        std                = np.std(x,axis=0)
        if save:
            parameters["mean"] = [mean.tolist()]
            parameters["std"]  = [std.tolist()]
    print("normalized shape:",mean.shape,std.shape)
    return (x - mean)/(std+1e-20)

def unnormalize(x):
    mean                       = np.array(parameters["mean"][0])
    std                        = np.array(parameters["std"][0])
    x                          = (x * std)+mean
    return x


def normalize_transitions(pres,sucs):
    """Normalize a dataset for image-based input format.
Normalization is performed across batches."""
    B, *F = pres.shape
    transitions = np.stack([pres,sucs], axis=1) # [B, 2, F]
    print(transitions.shape)

    normalized  = normalize(np.reshape(transitions, [-1, *F])) # [2BO, F]
    states      = normalized.reshape([-1, *F])
    transitions = states.reshape([-1, 2, *F])
    return transitions, states


def normalize_traces(traces):
    B, T, *F = traces.shape
    normalized = normalize(np.reshape(traces, [B * T, *F]))  # [B * T, F]
    traces = np.reshape(normalized, [B, T, *F])
    return traces