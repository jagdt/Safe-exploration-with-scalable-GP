# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:44:14 2025

@author: jjagdt
"""
from .defaultconfig_exploration import DefaultConfigExploration


class Config(DefaultConfigExploration):
    """
    Options class for the exploration setting
    """
    verbose = 2
    static_exploration = True

    solver_type = "safempc"

    # safempc
    beta_safety = 2.0
    n_safe = 1
    n_perf = 0
    r = 1

    # -- GP model
    gp_type = 'numpy'  # one of 'gpy', 'numpy', 'scalable'

    # -- Scalable GP specific parameters
    # Number of frequencies to use per dimension.
    n_frequencies = 5
    # Periods for each dimension.
    periods = [0.2*2, 1.5*2, 6.0*2]
    # Domain lengths for each dimension.
    domain_lengths = [0.2, 1.5, 6.0] # [dθ, θ, u]
    # Lengthscale multiple for the scalable GP to compute periods.
    lengthscale_multiple = 3.0

    # -- GP Bounds parameters
    # Whether to compute GP bounds. Otherwise uses constant ß.
    compute_bounds = True
    # Confidence level for the GP bounds
    delta = 0.05
    # Assumed RKHS norm of the true function
    rkhs_norm = None # None to compute automatically from posterior mean
    # Subgaussian noise bound
    R_subgaussian = [0.001, 0.0001] # Match the noise of the environment
    # Model mismatch offset
    projection_error = None # None to compute automatically

    def __init__(self):
        """ """
        super(Config, self).__init__(__file__)
