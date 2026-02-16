# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:44:14 2025

@author: jjagdt
"""
import numpy as np
from .defaultconfig_exploration import DefaultConfigExploration


class Config(DefaultConfigExploration):
    """
    Options class for continued exploration from a previous run
    """
    static_exploration = False
    n_safe = 2
    n_iterations = 10
    n_safe_samples = 500

    noise_std_dev = [5e-4, 5e-5]
    # environment
    env_name = "InvertedPendulum"
    env_options = dict()
    init_std = np.array([1.0, 0.2]) # standard deviation of random initial states for static exploration
    env_options["init_std"] = init_std
    env_options["plant_noise"] = np.array(noise_std_dev) ** 2
    env_options["max_deg"] = 20
    env_options["max_dtheta"] = 1.2
    env_options["max_dtheta_theta_0"] = 0.8
    solver_type = "safempc"
    pendulum_simple_constraints = False
    enable_objectives = False

    # -- GP model
    gp_type = 'numpy'  # one of 'gpy', 'numpy', 'scalable'
    use_global_opt_first = False
    domain_lengths = [6.0, 2.5, 2.0]  # [dθ, θ, u]
    update_model_interval = 1  # update GP with new data every n-th iteration, None to disable
    retrain_gp_interval = None # retrain the gp every n-th iteration, None to disable

    # -- Scalable GP specific parameters
    n_frequencies = 5
    periods = list(1.2 * np.array(domain_lengths))
    lengthscale_multiple = 3.0
    truncation_radius = [13.0, 14.0]
    truncation_target = [5e-6, 5e-7]

    # -- GP Bounds parameters
    compute_bounds = True
    plot_bounds = True
    delta = 0.05
    rkhs_norm = [20.0, 20.0]
    R_subgaussian = noise_std_dev
    projection_error = None

    # -- Reference GP parameters
    reference_gp = True
    reference_kern_types = ['sum_lin_rbf', 'sum_lin_rbf']
    reference_hyp = [
        {
            'rbf.lengthscale': np.array([7.8, 1.8, 6.2]),
            'rbf.variance': 0.00021,
            'linear.variances': np.array([0.0053, 0.0053, 0.011])
        },
        {
            'rbf.lengthscale': np.array([5.7, 6.3, 6.5]),
            'rbf.variance': 0.00075,
            'linear.variances': np.array([0.0068, 0.007, 0.011])
        }
    ]
    # Noise variances for the reference GP
    reference_noise_var = np.array(noise_std_dev) ** 2

    def __init__(self, file=None):
        """ """
        if file is not None:
            super(Config, self).__init__(file)
        else:
            super(Config, self).__init__(__file__)