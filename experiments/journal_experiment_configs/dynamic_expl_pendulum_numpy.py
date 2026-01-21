# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:44:14 2025

@author: jjagdt
"""
import numpy as np
from .defaultconfig_exploration import DefaultConfigExploration


class Config(DefaultConfigExploration):
    """
    Options class for the exploration setting
    """

    # -- GP model
    gp_type = 'scalable'  # one of 'gpy', 'numpy', 'scalable'
    # Whether to use global hyperparameter optimization first
    use_global_opt_first = False
    # Number of restarts for hyperparameter optimization
    n_restarts_gp = 1

    # -- Scalable GP specific parameters
    # Number of frequencies to use per dimension.
    n_frequencies = 5
    # Domain lengths for each dimension.
    domain_lengths = [7.0, 2.5, 2.0] # [dθ, θ, u]
    # Periods for each dimension.
    periods = list(1.5 * np.array(domain_lengths))
    # Lengthscale multiple for the scalable GP to compute periods.
    lengthscale_multiple = 3.0
    # Truncation radius for the scalable GP features
    truncation_radius = [9.0, 20.0]

    # -- GP Bounds parameters
    # Whether to compute GP bounds. Otherwise uses constant ß.
    compute_bounds = True
    # Whether to plot the GP bounds when plotting the model error and GP fit.
    plot_bounds = False
    # Confidence level for the GP bounds
    delta = 0.05
    # Assumed RKHS norm of the true function
    rkhs_norm = [20.0, 10.0]
    # Subgaussian noise bound
    R_subgaussian = [0.001, 0.0001] # Match the noise of the environment
    # Model mismatch offset
    projection_error = None # None to compute automatically

    # -- Reference GP parameters
    # Whether to use a reference GP with ground truth hyperparameters
    reference_gp = True
    # Kernel types for the reference GP
    reference_kern_types = ['sum_lin_rbf', 'sum_lin_rbf']
    # Hyperparameters of the reference GP
    reference_hyp = [
        {
            'rbf.lengthscale': np.array([0.53, 0.71, 0.61]),
            'rbf.variance': 0.0068,
            'linear.variances': np.array([0.94, 0.07, 0.0014])
        },
        {
            'rbf.lengthscale': np.array([0.52, 0.96, 1.08]),
            'rbf.variance': 1.05e-05,
            'linear.variances': np.array([0.01, 0.01, 0.01])
        }
    ]
    # Noise variances for the reference GP
    reference_noise_var = np.array([0.001, 0.0001]) ** 2

    def __init__(self):
        """ """
        super(Config, self).__init__(__file__)
