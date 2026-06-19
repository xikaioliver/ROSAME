import os.path
import numpy as np

from .common import *
from ..util.tuning import parameters
from .normalization import normalize_transitions, normalize_traces
import torch
from . import common

def load_data(data_loc, num_examples, **kwargs):
    # Load data from .npy file, and normalize images to [0, 1]
    img_traces = np.load(os.path.join(dl.__path__[0], "data", data_loc, "traces.npy")).astype(np.float32) / 255
    state_traces = np.load(os.path.join(dl.__path__[0], "data", data_loc, "states.npy")).astype(np.float32)
    action_traces = np.load(os.path.join(dl.__path__[0], "data", data_loc, "actions.npy")).astype(np.float32)
    picsize = img_traces[0, 0].shape
    print("loaded. picsize:", picsize)
    parameters["picsize"] = [list(picsize)]

    img_traces = normalize_traces(img_traces[:num_examples, 1:])
    img_traces = torch.tensor(img_traces).contiguous()
    state_traces = torch.tensor(state_traces[:num_examples]).contiguous()
    state_traces = torch.clamp(state_traces, min=0.0, max=1.0)
    action_traces = torch.tensor(action_traces[:num_examples]).contiguous()

    return img_traces, state_traces, action_traces

def rosame(args):
    img_traces, state_traces, action_traces = load_data(**vars(args))
    parameters["domain"] = args.domain
    ae = run(os.path.join("samples",common.sae_path), img_traces, state_traces, action_traces)


_parser = subparsers.add_parser('rosame_full',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help='ROSAME environment. Requires a domain name, and a data file')
_parser.add_argument('domain', help='choices of [blocks, gripper, logistics, hanoi, 8-puzzle]')
_parser.add_argument('data_loc', help='Name of trace and goal storage folder in dl/data/. Example: rosame_blocks_grid')
add_common_arguments(_parser, rosame)