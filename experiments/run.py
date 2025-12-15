#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:03:14 2017

@author: tkoller
"""

from typing import Optional, Tuple

import torch

from experiments import sacred_helper
from experiments.journal_experiment_configs.default_config import DefaultConfig
from safe_exploration.episode_runner import run_episodic
from safe_exploration.exploration_runner import run_exploration
from safe_exploration.rkhs_norm_estimation_runner import run_rkhs_norm_estimation
from safe_exploration.uncertainty_propagation_runner import run_uncertainty_propagation
from safe_exploration.utils_config import load_config, create_env, create_solver
from safe_exploration.utils_sacred import SacredAggregatedMetrics

ex = sacred_helper.get_experiment()


def check_config_conflicts(conf: DefaultConfig) -> Tuple[bool, str]:
    """ Check if there are conflicting options in the Config

    Parameters
    ----------
    conf: Config
        The config file

    Returns
    -------
    has_conflict: Bool
        True, if there is a conflict in the config
    conflict_str: String
        The error message

    """

    has_conflict = False
    conflict_str = ""
    if conf.task == "exploration" and not conf.solver_type in ("safempc", "safempc_cem"):
        return True, "Exploration task only allowed with safempc solver"
    elif conf.task == "uncertainty_propagation" and not conf.solver_type == "safempc":
        return True, "Uncertainty propagation task only allowed with safempc solver"
    elif conf.task == "rkhs_norm_estimation" and not conf.solver_type in ("safempc", "safempc_cem"):
        return True, "RKHS estimation task only allowed with safempc solver"

    return has_conflict, conflict_str


@ex.capture
def _run_scenario(_run, scenario_file: Optional[str], environment: Optional[str]):
    """ Run the specified scenario

    Parameters
    ----------
    args:
        The parsed arguments (see create_parser for details)
    """
    if scenario_file is not None and environment is None:
        conf = load_config(scenario_file)
    elif scenario_file is None and environment is not None:
        conf = load_config(_get_scenario_file_name(environment))
    else:
        raise ValueError('Must provide scenario file OR environment name')

    conf.add_sacred_config(_run.config)

    conflict, conflict_str = check_config_conflicts(conf)
    if conflict:
        raise ValueError("There are conflicting settings: {}".format(conflict_str))

    metrics = SacredAggregatedMetrics(_run)

    env = create_env(conf, conf.env_name, conf.env_options)

    task = conf.task
    if task == "exploration":
        run_exploration(conf, conf.visualize)
    elif task == "episode_setting":
        run_episodic(conf, metrics)
    elif task == "uncertainty_propagation":
        solver, safe_policy = create_solver(conf, env)
        run_uncertainty_propagation(env, solver, conf)
    elif task == "rkhs_norm_estimation":
        run_rkhs_norm_estimation(conf)
    else:
        raise ValueError(f"Unknown task: {task}")


def _get_scenario_file_name(environment_name: str) -> str:
    if environment_name == 'pendulum':
        return 'journal_experiment_configs/episodic_pendulum_cem.py'
    elif environment_name == 'lander':
        return 'journal_experiment_configs/episodic_lunar_lander_cem.py'
    else:
        raise ValueError(f'Unknown environment {environment_name}')


@ex.automain
def main():
    torch.set_default_dtype(torch.double)
    _run_scenario()
