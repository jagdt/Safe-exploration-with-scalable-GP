# -*- coding: utf-8 -*-

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

from .utils import generate_initial_samples, unavailable
from .utils_config import create_env, create_solver
from .visualization import plot_model_error_comparison

try:
	import matplotlib.pyplot as plt
	_has_matplotlib = True
except:
	_has_matplotlib = False


@unavailable(not _has_matplotlib, "matplotlib", conditionals=["visualize"])
def run_rkhs_norm_estimation(conf):
	"""Estimate RKHS norms of the true model error from safe initial data."""
	n_experiments = conf.n_experiments
	visualize = conf.visualize
	save_vis = conf.save_vis
	save_path = conf.save_path

	experiment_results = []

	for exp_idx in range(n_experiments):
		print(f"[RKHS] Starting experiment {exp_idx + 1}/{n_experiments}")
		env = create_env(conf, conf.env_name, conf.env_options)
		safempc, safe_policy = create_solver(conf, env)
		X, y = generate_initial_samples(env, conf, conf.relative_dynamics, safempc, 
										safe_policy)
		safempc.update_model(X, y, opt_hyp=conf.train_gp, reinitialize_solver=False)

		gp_model = getattr(safempc, "ssm", None)
		if gp_model is None:
			raise AttributeError("SafeMPC solver does not expose an SSM instance.")
		if not hasattr(gp_model, "estimate_true_rkhs_norm"):
			raise NotImplementedError(
				"Underlying GP model must implement 'estimate_true_rkhs_norm'."
			)

		x_train = gp_model.x_train
		y_train = gp_model.y_train

		rkhs_norms = gp_model.estimate_true_rkhs_norm(x_train, y_train)
		rkhs_norms = np.asarray(rkhs_norms, dtype=float)

		if visualize or save_vis:
			fig, ax = env.plot_safety_bounds(color="b")

			# plot the initial train set
			c_black = (0., 0., 0.)
			n_train, _ = np.shape(x_train)
			for i in range(n_train):
				ax = env.plot_state(ax, x_train[i, :env.n_s], color=c_black)

			if save_vis and save_path is not None:
				final_traj_plot_path = "{}/trajectory_final.png".format(save_path)
				fig.savefig(final_traj_plot_path, dpi=150, bbox_inches='tight')
				print(f"Saved final trajectory plot: {final_traj_plot_path}")			
				plt.close(fig)

			plot_model_error_comparison(safempc, env, save_dir=save_path, n_points=50, plot_bounds=conf.plot_bounds)

		result = {
			"experiment_idx": exp_idx,
			"n_samples": x_train.shape[0],
			"rkhs_norms": rkhs_norms,
			"kernel_types": getattr(gp_model, "kern_types", None),
		}
		print(
			f"[RKHS][Exp {result['experiment_idx']}] "
			f"samples={result['n_samples']}, "
			f"norms={rkhs_norms}"
		)
		experiment_results.append(result)

	_print_summary(experiment_results)


def _print_summary(results):
	"""Print aggregate statistics across experiments."""

	stacked = np.vstack([res["rkhs_norms"] for res in results])

	print("\n" + "="*60)
	print("RKHS NORM ESTIMATION SUMMARY")
	print("="*60)
	print(f"Number of experiments: {len(results)}")
	print(f"Samples per experiment: {results[0]['n_samples']}")
	print(f"Kernel types: {results[0]['kernel_types']}")
	print("-"*60)
	print(f"RKHS norms (mean ± std):")
	for dim_idx in range(stacked.shape[1]):
		mean_val = stacked[:, dim_idx].mean()
		std_val = stacked[:, dim_idx].std(ddof=0)
		min_val = stacked[:, dim_idx].min()
		max_val = stacked[:, dim_idx].max()
		print(f"  Dimension {dim_idx}: {mean_val:.6f} ± {std_val:.6f}  [min: {min_val:.6f}, max: {max_val:.6f}]")
	print("="*60 + "\n")

