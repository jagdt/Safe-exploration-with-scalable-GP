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

    init_mode = "safe_samples"
    n_safe_samples = 800
    init_std_initial_data = np.array([2.0, .5]) # standard deviation of initial safe samples
    init_m_initial_data = np.array([0., 0.])
    visualize_initial_samples = True

    env_options = dict()
    env_options["plant_noise"] = np.array([0.0, 0.0])
    env_options["max_deg"] = 60
    env_options["max_dtheta_theta_0"] = 4.0
    pendulum_simple_constraints = True

    n_experiments = 1
    n_iterations = 0

    # -- GP model
    gp_type = 'numpy'  # one of 'gpy', 'numpy', 'scalable'
    # Whether to use global hyperparameter optimization first
    use_global_opt_first = True

    # -- Scalable GP specific parameters
    # Number of frequencies to use per dimension.
    n_frequencies = 5
    # Domain lengths for each dimension.
    domain_lengths = [4.0, 2.5, 2.0] # [dθ, θ, u]
    # Periods for each dimension.
    periods = list(1.5 * np.array(domain_lengths))
    # Lengthscale multiple for the scalable GP to compute periods.
    lengthscale_multiple = 3.0
    # Truncation radius for the scalable GP features
    truncation_radius = 12.0

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
