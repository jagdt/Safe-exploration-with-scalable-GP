# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:54:37 2025

@author: jjagdt
"""
import numpy as np

from .dynamic_expl_pendulum_numpy import Config as ParentConfig


class Config(ParentConfig):
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
    env_options["max_dtheta_theta_0"] = 3.0
    pendulum_simple_constraints = True

    n_experiments = 1
    n_iterations = 0

    # -- GP model
    gp_type = 'numpy'  # one of 'gpy', 'numpy', 'scalable'
    # Whether to use global hyperparameter optimization first
    use_global_opt_first = False

    def __init__(self):
        """ """
        super(Config, self).__init__(file=__file__)
