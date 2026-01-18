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

    # -- GP model
    gp_type = 'scalable'  # one of 'gpy', 'numpy', 'scalable'

    # -- Scalable GP specific parameters
    # Number of frequencies to use per dimension.
    n_frequencies = 5
    # Periods for each dimension.
    periods = [3.0*2, 2.0*2, 2.0*2]
    # Domain lengths for each dimension.
    domain_lengths = [3.0, 2.0, 2.0] # [dθ, θ, u]
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
    rkhs_norm = 15.0
    # Subgaussian noise bound
    R_subgaussian = [0.001, 0.0001] # Match the noise of the environment
    # Model mismatch offset
    projection_error = None # None to compute automatically

    def __init__(self):
        """ """
        super(Config, self).__init__(__file__)
