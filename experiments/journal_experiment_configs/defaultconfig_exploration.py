# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""

import numpy as np

from .default_config import DefaultConfig


class DefaultConfigExploration(DefaultConfig):
    """
    Options class for the exploration setting
    """
    seed = 1
    verbose = 2

    # task options
    task = "exploration"  # don't change this
    static_exploration = True

    # environment
    env_name = "InvertedPendulum"
    env_options = dict()
    init_std = np.array([0.5, 0.2]) # standard deviation of random initial states for static exploration
    env_options["init_std"] = init_std
    env_options["plant_noise"] = np.array([0.001, 0.0001]) ** 2
    env_options["max_deg"] = 20
    env_options["max_dtheta"] = 1.2
    env_options["max_dtheta_theta_0"] = 0.8
    solver_type = "safempc"
    pendulum_simple_constraints = False
    enable_objectives = False

    # safempc
    beta_safety = 2.0
    n_safe = 2
    l_mu = None

    # Initial samples
    init_mode = "safe_samples"
    n_safe_samples = 1500
    init_randomized_safe_policy = True
    init_safe_policy_margin = 0.0
    init_safe_policy_exploration = 1.0
    init_safe_policy_min_width = 0.05
    c_max_probing_init = 3
    c_max_probing_next_state = 2
    init_std_initial_data = np.array([0.5, 0.2]) # standard deviation of initial safe samples
    init_m_initial_data = np.array([0., 0.])
    visualize_initial_samples = False

    lqr_wx_cost = np.diag([1., 2.])
    lqr_wu_cost = 25 * np.eye(1)
    init_ilqr = True
    str_cost_func = None

    # perf_traj_options
    n_perf = 0
    type_perf_traj = 'taylor'
    r = 1
    perf_has_fb = True

    # prior model: The prior model for the safempc approach
    # can be different from the true model (!)
    lin_prior = True
    prior_model = dict()
    prior_m = .149
    prior_b = 0.0
    prior_model["m"] = prior_m
    prior_model["b"] = prior_b

    # GP
    gp_dict_path = None
    gp_data_path = None  # None means no initial training data
    m = None  # subset of data of size m for training
    kern_types = ["sum_lin_rbf", "sum_lin_rbf"]
    train_gp = True  # train the gp initially?
    retrain_gp_interval = 1 # retrain the gp every n-th iteration, None to disable
    gp_hyp = None
    Z = None
    lin_trafo_gp_input = None
    gp_ns_in = 2
    gp_ns_out = 2
    relative_dynamics = False  ## This should be False

    gp_hyp = None
    # exploration
    n_experiments = 1
    n_iterations = 0
    n_restarts_optimizer = 10

    # general options
    verify_safety = False
    visualize = True
    save_results = True
    save_vis = True
    save_dir = None  # the directory such that the overall save location is save_path_base/save_dir/, if None uses timestamp
    save_path_base = "results_exploration"  # the directory such that the overall save location is save_path_base/save_dir/
    data_savename = None

    def __init__(self, file_path):
        self.file_path = file_path
