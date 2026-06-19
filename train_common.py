#!/usr/bin/env python3
import dl.main
from dl.util import *

# parameters : a dictionary of { name : [ *values ] or value }.
# If the value is a list, it is interpreted as a hyperparameter choice.
# If it is a non-list, single value, it is interpreted as a fixed hyperparameter.

# For each different combination of hyperparameters, a separate experiment will be run and recorded.
# Different choices of hyperparameters are considered as different experiments.
parameters = {
    # Network Architecture Hyperparameters
    # AAE Hyperparameters
    'aae_width': 1000,
    'aae_depth': 3,
    # Feature Encoder Hyperparameters
    'feature_dim': 512,
    # Symbol Net Hyperparameters
    'hidden_dim': 256,

    # Loss Weight Hyperparameters
    'lambda' :0.2, # Prior bias strength
    'gamma' :10, # Final step loss weight
    'beta_pred' :1, # Prediction (consistency) loss weight
    'beta_app': 1, # Applicability loss weight
    'beta_reconst': 0, # Reconstruction loss weight

    # Optimizer Hyperparameters
    'optimizer': "Adam",
    'lr': [1e-4],

    # Training Hyperparameters
    'epoch': 5000,
    'batch_size': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'pre_mip_epoch': 50,
    'mip_interval': 1,
    'mip_traces': 3,
    'pseudo_weight_decay': 0.99,
    'mip_time_limit': 60,  # in seconds

    'DL_to_MIP': [['state', 'action', 'model']],
    'MIP_to_DL': [['state', 'action', 'model']],

    'cp_type': 'mip-gurobi',
}

parameters_aux = {}

if __name__ == '__main__':
    dl.main.main(parameters)

