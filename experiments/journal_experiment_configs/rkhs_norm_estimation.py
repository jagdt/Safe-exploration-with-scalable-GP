# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:54:37 2025

@author: jjagdt
"""
import numpy as np

from .defaultconfig_exploration import DefaultConfigExploration


class Config(DefaultConfigExploration):
    """
    Options class for the exploration setting
    """
    task = "rkhs_norm_estimation"

    verbose = 2
    static_exploration = True

    solver_type = "safempc"

    # safempc
    beta_safety = 2.0
    n_safe = 2
    n_perf = 0
    r = 1

    init_std_initial_data = np.array([1.0, .8])
    init_m_initial_data = np.array([0., 0.])

    env_options = dict()
    init_std = np.array([.1, .1])
    env_options["init_std"] = init_std
    env_options["plant_noise"] = np.array([0.0, 0.0])

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
    # Whether to plot the GP bounds when plotting the model error and GP fit.
    plot_bounds = False
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
