# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:44:58 2017

@author: tkoller
"""
import functools
import warnings
from importlib import import_module
from os.path import abspath, exists, split

import numpy as np

from experiments.journal_experiment_configs.default_config import DefaultConfig
from . import safempc_cem
from .cautious_mpc import CautiousMPC
from .environments.environments import InvertedPendulum, CartPole, Environment
from .environments.ndpendulum import NDPendulum
from .environments.lunarlander import LunarLander
from .safempc_cem import CemSafeMPC
from .safempc_simple import SimpleSafeMPC
from .ssm_cem.gal_concrete_dropout import GalConcreteDropoutSSM
from .ssm_cem.gp_ssm_cem import GpCemSSM
from .ssm_cem.ssm_cem import CemSSM, JunkDimensionsSSM
# from .ssm_cem.dropout_ssm_cem import McDropoutSSM
from .utils import dlqr, unavailable
from .ssm_numpy.gp_numpy import NumpyGPModel
from .ssm_numpy.scalable_gp import ScalableGPModel

try:
    _has_ssm_gpy = True
    from safe_exploration.ssm_gpy.gaussian_process import SimpleGPModel
except:
    _has_ssm_gpy = False


def _create_cem_ssm(conf: DefaultConfig, env: Environment) -> CemSSM:
    if conf.cem_ssm == 'exact_gp':
        ssm_constructor =  functools.partial(GpCemSSM, conf=conf)
    elif conf.cem_ssm == 'mc_dropout':
        ssm_constructor =  functools.partial(McDropoutSSM, conf=conf)
    elif conf.cem_ssm == 'mc_dropout_gal':
        ssm_constructor =  functools.partial(GalConcreteDropoutSSM, conf=conf)
    else:
        raise ValueError(f'Unknown value for cem_ssm, {conf.cem_ssm}')

    # If we want to add junk dimensions then we need to wrap the ssm.
    if conf.junk_state_dimen > 0 or conf.junk_action_dimen > 0:
        return JunkDimensionsSSM(ssm_constructor, env.n_s, env.n_u, conf.junk_state_dimen, conf.junk_action_dimen)
    else:
        return ssm_constructor(state_dimen=env.n_s, action_dimen=env.n_u)


def _create_gp(conf, env):
    if conf.gp_type == "gpy":
        if not _has_ssm_gpy:
            raise ImportError(
                "gp_type='gpy' requires ssm_gpy to be installed. "
                "Either install the required dependencies or use gp_type='numpy' instead."
            )
        return SimpleGPModel(conf.gp_ns_out, conf.gp_ns_in, env.n_u, m=conf.m,
                         kern_types=conf.kern_types, Z=conf.Z)
    elif conf.gp_type == "numpy":
        return NumpyGPModel(conf.gp_ns_out, conf.gp_ns_in, env.n_u,
                         kern_types=conf.kern_types)
    elif conf.gp_type == "scalable":
        return ScalableGPModel(conf.gp_ns_out, conf.gp_ns_in, env.n_u,
                         kern_types=conf.kern_types, n_frequencies=conf.n_frequencies,
                         periods=conf.periods, domain_lengths=conf.domain_lengths,
                         lengthscale_multiple=conf.lengthscale_multiple)
        raise NotImplementedError("Scalable GP not implemented for SimpleSafeMPC")
    else:
        raise ValueError(f"Unknown gp_type: {conf.gp_type}")

def create_solver(conf, env: Environment):
    """ Create a solver from a set of options and environment information"""
    lin_model = None
    lin_trafo_gp_input = conf.lin_trafo_gp_input

    h_mat_safe, h_safe, h_mat_obs, h_obs = env.get_safety_constraints(normalize=True)

    warnings.warn("Normalization of constraints may be wrong!")

    a_true, b_true = env.linearize_discretize()

    safe_policy = None
    if conf.lin_prior:
        a_prior, b_prior = get_prior_model_from_conf(conf, env)
        lin_model = (a_prior, b_prior)
        # by default takes identity as prior. Need to define safe_policy
    wx_cost = conf.lqr_wx_cost
    wu_cost = conf.lqr_wu_cost

    q = wx_cost
    r = wu_cost
    k_lqr, _, _ = dlqr(a_true, b_true, q, r)
    k_fb = -k_lqr
    safe_policy = lambda x: np.dot(k_fb, x)

    dt = env.dt
    ctrl_bounds = np.hstack(
        (np.reshape(env.u_min_norm, (-1, 1)), np.reshape(env.u_max_norm, (-1, 1))))

    env_opts_safempc = dict()

    env_opts_safempc["h_mat_safe"] = h_mat_safe
    env_opts_safempc["h_safe"] = h_safe
    env_opts_safempc["lin_model"] = lin_model
    env_opts_safempc["ctrl_bounds"] = ctrl_bounds
    env_opts_safempc["safe_policy"] = safe_policy
    env_opts_safempc["h_mat_obs"] = h_mat_obs
    env_opts_safempc["h_obs"] = h_obs
    env_opts_safempc["dt"] = env.dt
    env_opts_safempc["lin_trafo_gp_input"] = lin_trafo_gp_input

    if conf.solver_type == "safempc":
        # the environment options needed for the safempc algorithm
        env_opts_safempc["l_mu"] = env.l_mu
        env_opts_safempc["l_sigma"] = env.l_sigm

        perf_opts_safempc = dict()
        perf_opts_safempc["type_perf_traj"] = conf.type_perf_traj
        perf_opts_safempc["n_perf"] = conf.n_perf
        perf_opts_safempc["r"] = conf.r
        perf_opts_safempc["perf_has_fb"] = conf.perf_has_fb

        gp = _create_gp(conf, env)
        solver = SimpleSafeMPC(conf.n_safe, gp, env_opts_safempc, wx_cost, wu_cost,
                               beta_safety=conf.beta_safety,
                               safe_policy=safe_policy,
                               opt_perf_trajectory=perf_opts_safempc,
                               lin_trafo_gp_input=lin_trafo_gp_input, verbosity=conf.verbose)
    elif conf.solver_type == "safempc_cem":
        ssm = _create_cem_ssm(conf, env)
        constraints = safempc_cem.construct_constraints(conf, env)
        solver = CemSafeMPC(ssm, constraints=constraints, env=env, conf=conf, opt_env=env_opts_safempc,
                            wx_feedback_cost=wx_cost, wu_feedback_cost=wu_cost, beta_safety=conf.cem_beta_safety,
                            safe_policy=safe_policy)
    elif conf.solver_type == "cautious_mpc":
        T = conf.T

        gp = _create_gp(conf, env)
        solver = CautiousMPC(T, gp, env_opts_safempc, conf.beta_safety,
                             lin_trafo_gp_input=lin_trafo_gp_input, perf_trajectory=conf.type_perf_traj, k_fb=k_fb)
    else:
        raise ValueError("Unknown solver type: {}".format(conf.solver_type))

    # mpc_control = SafeMPC(n_safe, gp, env_opts_safempc, wx_cost, wu_cost,beta_safety = conf.beta_safety,
    #             ilqr_init = conf.ilqr_init, lin_model = lin_model, ctrl_bounds = ctrl_bounds,
    #             safe_policy = safe_policy, opt_perf_trajectory = perf_opts_safempc,lin_trafo_gp_input = lin_trafo_gp_input)

    return solver, safe_policy


def create_env(conf, env_name, env_options_dict=None):
    """ Given a set of options, create an environment """
    if env_options_dict is None:
        env_options_dict = dict()

    if env_name == "InvertedPendulum":
        if not hasattr(conf, 'pendulum_dimensions') or conf.pendulum_dimensions == 2:
            return InvertedPendulum(**env_options_dict, simple_constraints=conf.pendulum_simple_constraints,
                                    enable_objectives=conf.enable_objectives)
        elif conf.pendulum_dimensions > 2:
            return NDPendulum(**env_options_dict)
        else:
            raise ValueError(f'Unknown pendulum dimensions {conf.pendulum_dimensions}')

    elif env_name == "CartPole":
        return CartPole(**env_options_dict)

    elif env_name == "LunarLander":
        return LunarLander(conf)

    else:
        raise NotImplementedError("Unknown environment: {}".format(env_name))


def get_prior_model_from_conf(conf, env_true):
    """ Get prior model from config"""
    if conf.lin_prior:
        # unless specified otherwise, use the same normalization as for the true model
        if not "norm_x" in conf.prior_model:
            conf.prior_model["norm_x"] = env_true.norm[0]
        if not "norm_u" in conf.prior_model:
            conf.prior_model["norm_u"] = env_true.norm[1]

        env_prior = create_env(conf, conf.env_name, conf.prior_model)

        a_prior, b_prior = env_prior.linearize_discretize()

    else:
        a_prior, b_prior = (np.eye(env_true.n_s), np.zeros((env_true.n_s, env_true.n_u)))

    return a_prior, b_prior


def get_model_options_from_conf(conf, env):
    """ Utility function to create a gp options dict from the config class"""

    # There already is a gp_dict ready to use
    if not conf.gp_dict_path is None:
        return np.load(conf.gp_dict_path)

    # neither a gp_dict_path nor a gp_data_path exists -> return None
    gp_dict = dict()

    a_prior, b_prior = get_prior_model_from_conf(conf, env)

    ab_prior = np.hstack((a_prior, b_prior))
    prior_model = lambda z: np.dot(z, ab_prior.T)
    gp_dict["prior_model"] = prior_model

    gp_dict["data_path"] = conf.gp_data_path
    gp_dict["m"] = conf.m
    gp_dict["kern_types"] = conf.kern_types
    gp_dict["hyp"] = conf.gp_hyp
    gp_dict["train"] = conf.train_gp
    gp_dict["n_s_in"] = conf.gp_ns_in
    gp_dict["n_s_out"] = conf.gp_ns_out
    gp_dict["n_u"] = conf.gp_nu
    gp_dict["Z"] = conf.Z

    return gp_dict


def load_config(conf_path: str) -> DefaultConfig:
    if not exists(abspath(conf_path)):
        raise ValueError
    else:

        file_path, filename = split(conf_path)
        module_name = filename.split('.')[0]
        module_name_path = file_path + '.' + module_name
        configuration = import_module(module_name_path)
        conf = configuration.Config()

        return conf
