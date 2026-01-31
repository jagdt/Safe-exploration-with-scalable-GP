#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot model error analysis for safe exploration experiments

Uses actual trained GP and environment from MPC framework to compare:
1. True model error: f_true(x,u) - f_prior(x,u)
2. GP learned error: GP(x,u)
3. GP uncertainty: ±2σ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
import pickle

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'figure.figsize': (8, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

# RWTH Aachen University corporate colors
RWTH_BLUE = '#00549F'
RWTH_BLACK = '#000000'
RWTH_MAGENTA = '#E30066'
RWTH_TURQUOISE = '#0098A1'
RWTH_GREEN = '#57AB27'
RWTH_ORANGE = '#F6A800'
RWTH_RED = '#CC071E'
RWTH_BORDEAUX = '#A11035'
RWTH_PURPLE = '#612158'
RWTH_LIGHT_BLUE = '#8EBAE5'
RWTH_GRAY = '#9C9E9F'

# Create RWTH trajectory colormap (same as in exploration_runner.py)
colors_list = [RWTH_LIGHT_BLUE, RWTH_BLUE, RWTH_MAGENTA]
n_bins = 256
RWTH_CMAP = LinearSegmentedColormap.from_list('rwth_trajectory', colors_list, N=n_bins)


def compute_true_model_error(safempc, env, states, actions):
    """Compute true model error using SafeMPC's environments
    
    Parameters
    ----------
    safempc : SimpleSafeMPC
        The MPC controller with environment and prior model
    states : ndarray [N x n_s]
        States to evaluate (normalized)
    actions : ndarray [N x n_u]
        Actions to evaluate (normalized)
        
    Returns
    -------
    true_error : ndarray [N x n_s]
        True model error (normalized)
    """
    N = states.shape[0]
    n_s = safempc.n_s
    
    true_next = np.zeros((N, n_s))
    
    for i in range(N):
        next_state_norm, _ = env.simulate_onestep(states[i], actions[i])
        true_next[i] = next_state_norm
    
    prior_next = safempc.eval_prior(states, actions)
    
    true_error = true_next - prior_next
    
    return true_error


def plot_model_error_comparison(safempc, env, save_dir=None, n_points=30, plot_bounds=False, n_initial_samples=0):
    """Create comprehensive model error comparison plots
    
    Parameters
    ----------
    safempc : SimpleSafeMPC
        The MPC controller with trained GP
    save_dir : str, optional
        Directory to save plots. If None, plots are not saved.
    n_points : int
        Number of points per dimension for grid
    plot_bounds : bool
        Whether to plot confidence bounds
    n_initial_samples : int
        Number of initial samples (will be plotted in gray, rest with RWTH colormap)
    """
    
    if save_dir is not None:
        save_dir = os.path.join(save_dir, 'model_error_plots')
        os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Model Error Visualization")
    print("=" * 60)
    
    print(f"\nGP Model: {type(safempc.ssm).__name__}")
    print(f"  Training samples: {safempc.ssm.x_train.shape[0]}")
    print(f"  State dimensions: {safempc.n_s}")
    print(f"  Control dimensions: {safempc.n_u}")
    
    print(f"\nGenerating test grid ({n_points} points per dimension)...")
    
    x_train = safempc.ssm.x_train
    state_min = x_train[:, :safempc.n_s].min(axis=0)
    state_max = x_train[:, :safempc.n_s].max(axis=0)
    action_min = x_train[:, safempc.n_s:].min(axis=0)
    action_max = x_train[:, safempc.n_s:].max(axis=0)
    
    print(f"  State range: {state_min} to {state_max}")
    print(f"  Action range: {action_min} to {action_max}")

    state_range = state_max - state_min
    state_min -= 0.2 * state_range
    state_max += 0.2 * state_range
    
    action_range = action_max - action_min
    action_min -= 0.2 * action_range
    action_max += 0.2 * action_range
    
    if safempc.n_s == 2:  # Inverted pendulum
        # 1D slices: vary each dimension individually
        # 1D slice 1: vary u, fix dtheta=0, theta=0
        u_1d = np.linspace(action_min[0], action_max[0], n_points)
        states_1d_u = np.column_stack([
            np.zeros(n_points),  # dtheta = 0
            np.zeros(n_points)   # theta = 0
        ])
        actions_1d_u = u_1d[:, np.newaxis]
        
        # 1D slice 2: vary theta, fix dtheta=0, u=0
        theta_1d = np.linspace(state_min[1], state_max[1], n_points)
        states_1d_theta = np.column_stack([
            np.zeros(n_points),  # dtheta = 0
            theta_1d             # theta varies
        ])
        actions_1d_theta = np.zeros((n_points, 1))
        
        # 1D slice 3: vary dtheta, fix theta=0, u=0
        dtheta_1d = np.linspace(state_min[0], state_max[0], n_points)
        states_1d_dtheta = np.column_stack([
            dtheta_1d,           # dtheta varies
            np.zeros(n_points)   # theta = 0
        ])
        actions_1d_dtheta = np.zeros((n_points, 1))
        
        # 2D grid 1: vary theta and u, fix dtheta=0
        theta_2d = np.linspace(state_min[1], state_max[1], n_points)
        u_2d = np.linspace(action_min[0], action_max[0], n_points)
        theta_grid, u_grid = np.meshgrid(theta_2d, u_2d)
        states_2d_theta_u = np.column_stack([
            np.zeros_like(theta_grid.ravel()),
            theta_grid.ravel()
        ])
        actions_2d_theta_u = u_grid.ravel()[:, np.newaxis]
        
        # 2D grid 2: vary dtheta and u, fix theta=0
        dtheta_2d = np.linspace(state_min[0], state_max[0], n_points)
        dtheta_grid, u_grid_2 = np.meshgrid(dtheta_2d, u_2d)
        states_2d_dtheta_u = np.column_stack([
            dtheta_grid.ravel(),
            np.zeros_like(dtheta_grid.ravel())
        ])
        actions_2d_dtheta_u = u_grid_2.ravel()[:, np.newaxis]
        
        # 2D grid 3: vary dtheta and theta, fix u=0
        dtheta_theta_grid, theta_dtheta_grid = np.meshgrid(dtheta_2d, theta_2d)
        states_2d_dtheta_theta = np.column_stack([
            dtheta_theta_grid.ravel(),
            theta_dtheta_grid.ravel()
        ])
        actions_2d_dtheta_theta = np.zeros((n_points * n_points, 1))
        
        dim_names = ['dθ', 'θ', 'u']
        error_names = ['Δ(dθ)', 'Δ(θ)']
        
    else:
        raise NotImplementedError("Model error plotting not implemented for n_s != 2")

    print("Computing true model error...")
    true_error_1d_u = compute_true_model_error(safempc, env, states_1d_u, actions_1d_u)
    true_error_1d_theta = compute_true_model_error(safempc, env, states_1d_theta, actions_1d_theta)
    true_error_1d_dtheta = compute_true_model_error(safempc, env, states_1d_dtheta, actions_1d_dtheta)

    true_error_2d_theta_u = compute_true_model_error(safempc, env, states_2d_theta_u, actions_2d_theta_u)
    true_error_2d_dtheta_u = compute_true_model_error(safempc, env, states_2d_dtheta_u, actions_2d_dtheta_u)
    true_error_2d_dtheta_theta = compute_true_model_error(safempc, env, states_2d_dtheta_theta, actions_2d_dtheta_theta)

    print("Getting GP predictions...")
    
    if hasattr(safempc, 'lin_trafo_gp_input'):
        from numpy import dot as mtimes
        states_1d_u_trafo = mtimes(states_1d_u, safempc.lin_trafo_gp_input.T)
        test_inputs_1d_u = np.hstack([states_1d_u_trafo, actions_1d_u])
        
        states_1d_theta_trafo = mtimes(states_1d_theta, safempc.lin_trafo_gp_input.T)
        test_inputs_1d_theta = np.hstack([states_1d_theta_trafo, actions_1d_theta])
        
        states_1d_dtheta_trafo = mtimes(states_1d_dtheta, safempc.lin_trafo_gp_input.T)
        test_inputs_1d_dtheta = np.hstack([states_1d_dtheta_trafo, actions_1d_dtheta])
        
        states_2d_theta_u_trafo = mtimes(states_2d_theta_u, safempc.lin_trafo_gp_input.T)
        states_2d_dtheta_u_trafo = mtimes(states_2d_dtheta_u, safempc.lin_trafo_gp_input.T)
        states_2d_dtheta_theta_trafo = mtimes(states_2d_dtheta_theta, safempc.lin_trafo_gp_input.T)
        test_inputs_2d_theta_u = np.hstack([states_2d_theta_u_trafo, actions_2d_theta_u])
        test_inputs_2d_dtheta_u = np.hstack([states_2d_dtheta_u_trafo, actions_2d_dtheta_u])
        test_inputs_2d_dtheta_theta = np.hstack([states_2d_dtheta_theta_trafo, actions_2d_dtheta_theta])
    else:
        test_inputs_1d_u = np.hstack([states_1d_u, actions_1d_u])
        test_inputs_1d_theta = np.hstack([states_1d_theta, actions_1d_theta])
        test_inputs_1d_dtheta = np.hstack([states_1d_dtheta, actions_1d_dtheta])
    
        test_inputs_2d_theta_u = np.hstack([states_2d_theta_u, actions_2d_theta_u])
        test_inputs_2d_dtheta_u = np.hstack([states_2d_dtheta_u, actions_2d_dtheta_u])
        test_inputs_2d_dtheta_theta = np.hstack([states_2d_dtheta_theta, actions_2d_dtheta_theta])
    
    gp_mean_1d_u, gp_std_1d_u = safempc.ssm.predict(test_inputs_1d_u)
    gp_mean_1d_theta, gp_std_1d_theta = safempc.ssm.predict(test_inputs_1d_theta)
    gp_mean_1d_dtheta, gp_std_1d_dtheta = safempc.ssm.predict(test_inputs_1d_dtheta)

    gp_mean_2d_theta_u, gp_std_2d_theta_u = safempc.ssm.predict(test_inputs_2d_theta_u)
    gp_mean_2d_dtheta_u, gp_std_2d_dtheta_u = safempc.ssm.predict(test_inputs_2d_dtheta_u)
    gp_mean_2d_dtheta_theta, gp_std_2d_dtheta_theta = safempc.ssm.predict(test_inputs_2d_dtheta_theta)
    
    print("Generating plots...")
    
    x_train_gp = safempc.ssm.x_train
    y_train = safempc.ssm.y_train
    n_s = safempc.n_s
    
    if hasattr(safempc, 'lin_trafo_gp_input'):
        n_u = safempc.n_u
        states_train_trafo = x_train_gp[:, :n_s]
        actions_train = x_train_gp[:, n_s:]
        
        trafo_inv = np.linalg.inv(safempc.lin_trafo_gp_input.T)
        states_train_orig = mtimes(states_train_trafo, trafo_inv)
        
        x_train = np.hstack([states_train_orig, actions_train])
    else:
        states_train_orig = x_train_gp[:, :n_s]
        actions_train = x_train_gp[:, n_s:]
        x_train = x_train_gp
    
    
    beta = 2.0
    if hasattr(safempc.ssm, 'beta_safety'):
        beta = safempc.ssm.beta_safety
    elif hasattr(safempc, 'beta_safety'):
        beta = safempc.beta_safety
        
    proj_error = 0.0
    if hasattr(safempc.ssm, 'projection_error_per_dim'):
        proj_error = safempc.ssm.projection_error_per_dim
    elif hasattr(safempc, 'projection_error_per_dim'):
        proj_error = safempc.projection_error_per_dim
    
    if proj_error is None:
        proj_error = 0.0
        
    def get_param_for_dim(param, dim, n_s):
        if np.size(param) == 1:
            return float(param)
        else:
            param = np.asarray(param).flatten()
            if param.size == n_s:
                return param[dim]
            warnings.warn(f"Parameter size {param.size} does not match n_s {n_s}, using first element")
            return param[0]
            
    print(f"Using beta={beta}, projection_error={proj_error}")

    save_path = None

    # Plot 1D slice varying u
    for dim in range(safempc.n_s):
        beta_dim = get_param_for_dim(beta, dim, safempc.n_s)
        proj_dim = get_param_for_dim(proj_error, dim, safempc.n_s)
        
        if save_dir is not None:
            save_path = os.path.join(save_dir, f'model_error_1d_u_dim{dim}.png')
        plot_1d_comparison(
            states_1d_u, actions_1d_u, true_error_1d_u, gp_mean_1d_u, gp_std_1d_u,
            state_dim=dim, vary_dim=2,  # u is at index 2
            dim_names=dim_names, error_names=error_names,
            save_path=save_path,
            x_train=x_train, y_train=y_train,
            beta=beta_dim, proj_error=proj_dim,
            plot_bounds=plot_bounds,
            n_initial_samples=n_initial_samples
        )
    
    # Plot 1D slice varying theta
    for dim in range(safempc.n_s):
        beta_dim = get_param_for_dim(beta, dim, safempc.n_s)
        proj_dim = get_param_for_dim(proj_error, dim, safempc.n_s)
        
        if save_dir is not None:
            save_path = os.path.join(save_dir, f'model_error_1d_theta_dim{dim}.png')
        plot_1d_comparison(
            states_1d_theta, actions_1d_theta, true_error_1d_theta, gp_mean_1d_theta, gp_std_1d_theta,
            state_dim=dim, vary_dim=1,  # theta is at index 1
            dim_names=dim_names, error_names=error_names,
            save_path=save_path,
            x_train=x_train, y_train=y_train,
            beta=beta_dim, proj_error=proj_dim,
            plot_bounds=plot_bounds,
            n_initial_samples=n_initial_samples
        )
    
    # Plot 1D slice varying dtheta
    for dim in range(safempc.n_s):
        beta_dim = get_param_for_dim(beta, dim, safempc.n_s)
        proj_dim = get_param_for_dim(proj_error, dim, safempc.n_s)
        
        if save_dir is not None:
            save_path = os.path.join(save_dir, f'model_error_1d_dtheta_dim{dim}.png')
        plot_1d_comparison(
            states_1d_dtheta, actions_1d_dtheta, true_error_1d_dtheta, gp_mean_1d_dtheta, gp_std_1d_dtheta,
            state_dim=dim, vary_dim=0,  # dtheta is at index 0
            dim_names=dim_names, error_names=error_names,
            save_path=save_path,
            x_train=x_train, y_train=y_train,
            beta=beta_dim, proj_error=proj_dim,
            plot_bounds=plot_bounds,
            n_initial_samples=n_initial_samples
        )
    
    # Plot theta vs u
    for dim in range(safempc.n_s):
        if save_dir is not None:
            save_path = os.path.join(save_dir, f'model_error_2d_theta_u_dim{dim}.png')
        plot_2d_comparison(
            states_2d_theta_u, actions_2d_theta_u, true_error_2d_theta_u, 
            gp_mean_2d_theta_u, gp_std_2d_theta_u,
            state_dim=dim, 
            vary_dims=(1, safempc.n_s),
            n_points=n_points,
            dim_names=dim_names, error_names=error_names,
            save_path=save_path,
            x_train=x_train, y_train=y_train,
            n_initial_samples=n_initial_samples
        )
    
    # Plot dtheta vs u
    for dim in range(safempc.n_s):
        if save_dir is not None:
            save_path = os.path.join(save_dir, f'model_error_2d_dtheta_u_dim{dim}.png')
        plot_2d_comparison(
            states_2d_dtheta_u, actions_2d_dtheta_u, true_error_2d_dtheta_u, 
            gp_mean_2d_dtheta_u, gp_std_2d_dtheta_u,
            state_dim=dim, 
            vary_dims=(0, safempc.n_s),
            n_points=n_points,
            dim_names=dim_names, error_names=error_names,
            save_path=save_path,
            x_train=x_train, y_train=y_train,
            n_initial_samples=n_initial_samples
        )
    
    # Plot dtheta vs theta
    for dim in range(safempc.n_s):
        if save_dir is not None:
            save_path = os.path.join(save_dir, f'model_error_2d_dtheta_theta_dim{dim}.png')
        plot_2d_comparison(
            states_2d_dtheta_theta, actions_2d_dtheta_theta, true_error_2d_dtheta_theta, 
            gp_mean_2d_dtheta_theta, gp_std_2d_dtheta_theta,
            state_dim=dim, 
            vary_dims=(0, 1),
            n_points=n_points,
            dim_names=dim_names, error_names=error_names,
            save_path=save_path,
            x_train=x_train, y_train=y_train,
            n_initial_samples=n_initial_samples
        )

    # 3D scatter of training data highlighting residual mismatch
    gp_mean_train, _ = safempc.ssm.predict(x_train_gp)
    true_error_train = compute_true_model_error(safempc, env, states_train_orig, actions_train)
    
    if save_dir is not None:
        scatter_save_path = os.path.join(save_dir, 'model_error_training_scatter.png')
    else:
        scatter_save_path = None
    
    plot_training_error_scatter(
        states_train_orig,
        actions_train,
        true_error_train,
        gp_mean_train,
        dim_names,
        error_names,
        save_path=scatter_save_path,
        n_initial_samples=n_initial_samples
    )
    
    print_statistics(true_error_2d_theta_u, gp_mean_2d_theta_u, gp_std_2d_theta_u, error_names)
    
    print(f"\n{'='*60}")
    print(f"All plots displayed")
    print(f"{'='*60}\n")


def plot_1d_comparison(states, actions, true_error, gp_mean, gp_std,
                       state_dim, vary_dim, dim_names, error_names, save_path, x_train=None, y_train=None,
                       beta=2.0, proj_error=0.0, plot_bounds=False, n_initial_samples=0):
    """Plot 1D comparison of true vs predicted model error"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Get x-axis data
    if vary_dim < states.shape[1]:
        x_data = states[:, vary_dim]
    else:
        x_data = actions[:, vary_dim - states.shape[1]]
    
    # Sort for cleaner lines
    sort_idx = np.argsort(x_data)
    x_sorted = x_data[sort_idx]
    true_sorted = true_error[sort_idx, state_dim]
    gp_sorted = gp_mean[sort_idx, state_dim]
    gp_std_sorted = gp_std[sort_idx, state_dim]
    
    # Plot: True vs GP prediction
    ax.plot(x_sorted, true_sorted, color=RWTH_BLUE, linewidth=2.5, 
            label='True model error', alpha=0.9, zorder=3)
    ax.plot(x_sorted, gp_sorted, color=RWTH_MAGENTA, linestyle='--', linewidth=2.5, 
            label='GP prediction', alpha=0.9, zorder=3)
    ax.fill_between(x_sorted,
                    gp_sorted - 2*gp_std_sorted,
                    gp_sorted + 2*gp_std_sorted,
                    color=RWTH_MAGENTA, alpha=0.15, label=r'GP $\pm 2\sigma$', zorder=2)
    
    # Plot Confidence Bound (Safety Bound)
    if plot_bounds:
        bound = beta * gp_std_sorted + proj_error
        ax.plot(x_sorted, gp_sorted + bound, color=RWTH_BLACK, linestyle='-.', 
                linewidth=1.5, label='Safety bound', alpha=0.7, zorder=3)
        ax.plot(x_sorted, gp_sorted - bound, color=RWTH_BLACK, linestyle='-.', 
                linewidth=1.5, alpha=0.7, zorder=3)
        ax.fill_between(x_sorted, gp_sorted - bound, gp_sorted + bound,
                        color=RWTH_GRAY, alpha=0.2, label='Safety region', zorder=1)
    
    # Plot training points with time-ordered colorbar
    if x_train is not None and y_train is not None:
        train_x = x_train[:, vary_dim]
        train_y = y_train[:, state_dim]
        n_train = len(train_x)
        
        # Separate initial samples from exploration samples
        if n_initial_samples > 0 and n_initial_samples < n_train:
            # Plot initial samples in gray
            ax.scatter(train_x[:n_initial_samples], train_y[:n_initial_samples], 
                      c=RWTH_GRAY, s=60, alpha=0.3, edgecolors=RWTH_BLACK, linewidth=0.8, 
                      label='Initial samples', zorder=4)
            
            # Plot exploration samples with RWTH colormap
            n_exploration = n_train - n_initial_samples
            time_colors = np.arange(n_exploration)
            scatter = ax.scatter(train_x[n_initial_samples:], train_y[n_initial_samples:], 
                               c=time_colors, cmap=RWTH_CMAP, 
                               s=60, alpha=0.7, edgecolors=RWTH_BLACK, linewidth=0.8, 
                               label='Exploration samples', zorder=5)
            
            # Add colorbar for exploration samples
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
            cbar.set_label('Exploration step', fontsize=12, rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=10)
        else:
            # All samples with colormap (no distinction)
            time_colors = np.arange(n_train)
            scatter = ax.scatter(train_x, train_y, c=time_colors, cmap=RWTH_CMAP, 
                               s=60, alpha=0.7, edgecolors=RWTH_BLACK, linewidth=0.8, 
                               label='Training data', zorder=5)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
            cbar.set_label('Sample order', fontsize=12, rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=10)
    
    ax.axhline(0, color=RWTH_BLACK, linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Improved axis labels with LaTeX
    xlabel_map = {'dθ': r'$\dot{\vartheta}$ [rad/s]', 'θ': r'$\vartheta$ [rad]', 'u': r'$u$ [Nm]'}
    ylabel_map = {'Δ(dθ)': r'$\Delta \dot{\vartheta}$ [rad/s]', 'Δ(θ)': r'$\Delta \vartheta$ [rad]'}
    
    ax.set_xlabel(xlabel_map.get(dim_names[vary_dim], dim_names[vary_dim]), fontsize=14)
    ax.set_ylabel(ylabel_map.get(error_names[state_dim], error_names[state_dim]), fontsize=14)
    ax.set_title(f'Model Error Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        # Also save as PDF
        # pdf_path = save_path.replace('.png', '.pdf')
        # plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {error_names[state_dim]} (1D) -> {save_path}")
        plt.close(fig)
    else:
        plt.show()
        print(f"  Displayed: {error_names[state_dim]} (1D)")


def plot_2d_comparison(states, actions, true_error, gp_mean, gp_std,
                       state_dim, vary_dims, n_points, dim_names, error_names, save_path, x_train=None, y_train=None, n_initial_samples=0):
    """Plot 2D heatmaps comparing true vs predicted model error"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Get x, y data
    x_dim, y_dim = vary_dims
    if x_dim < states.shape[1]:
        x_data = states[:, x_dim]
    else:
        x_data = actions[:, x_dim - states.shape[1]]
    
    if y_dim < states.shape[1]:
        y_data = states[:, y_dim]
    else:
        y_data = actions[:, y_dim - states.shape[1]]
    
    # Reshape to grid
    X = x_data.reshape(n_points, n_points)
    Y = y_data.reshape(n_points, n_points)
    Z_true = true_error[:, state_dim].reshape(n_points, n_points)
    Z_gp = gp_mean[:, state_dim].reshape(n_points, n_points)
    Z_std = gp_std[:, state_dim].reshape(n_points, n_points)
    
    # Determine common color scale for error plots
    vmin = min(Z_true.min(), Z_gp.min())
    vmax = max(Z_true.max(), Z_gp.max())
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)
    
    # Axis label mappings
    label_map = {'dθ': r'$\dot{\vartheta}$ [rad/s]', 'θ': r'$\vartheta$ [rad]', 'u': r'$u$ [Nm]'}
    error_label_map = {'Δ(dθ)': r'$\Delta \dot{\vartheta}$', 'Δ(θ)': r'$\Delta \vartheta$'}
    
    # Plot 1: True model error
    im1 = axes[0].contourf(X, Y, Z_true, levels=levels, cmap='RdBu_r', norm=norm)
    axes[0].contour(X, Y, Z_true, levels=levels, colors='black', linewidths=0.3, alpha=0.3)
    
    if x_train is not None and y_train is not None:
        train_x = x_train[:, x_dim]
        train_y = x_train[:, y_dim]
        n_train = len(train_x)
        
        # Separate initial samples from exploration samples
        if n_initial_samples > 0 and n_initial_samples < n_train:
            # Plot initial samples in gray
            axes[0].scatter(train_x[:n_initial_samples], train_y[:n_initial_samples], 
                          c=RWTH_GRAY, s=40, alpha=0.3, edgecolors=RWTH_BLACK, linewidth=0.8, zorder=4)
            
            # Plot exploration samples with RWTH colormap
            n_exploration = n_train - n_initial_samples
            time_colors = np.arange(n_exploration)
            scatter1 = axes[0].scatter(train_x[n_initial_samples:], train_y[n_initial_samples:], 
                                      c=time_colors, cmap=RWTH_CMAP, 
                                      s=40, alpha=0.8, edgecolors=RWTH_BLACK, linewidth=0.8, zorder=5)
        else:
            # All samples with colormap
            time_colors = np.arange(n_train)
            scatter1 = axes[0].scatter(train_x, train_y, c=time_colors, cmap=RWTH_CMAP, 
                                      s=40, alpha=0.8, edgecolors=RWTH_BLACK, linewidth=0.8, zorder=5)
    
    axes[0].set_xlabel(label_map.get(dim_names[x_dim], dim_names[x_dim]), fontsize=14)
    axes[0].set_ylabel(label_map.get(dim_names[y_dim], dim_names[y_dim]), fontsize=14)
    axes[0].set_title(f'True Error: {error_label_map.get(error_names[state_dim], error_names[state_dim])}', 
                     fontsize=15, fontweight='bold', pad=10)
    axes[0].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Plot 2: GP predicted error
    im2 = axes[1].contourf(X, Y, Z_gp, levels=levels, cmap='RdBu_r', norm=norm)
    axes[1].contour(X, Y, Z_gp, levels=levels, colors='black', linewidths=0.3, alpha=0.3)
    
    if x_train is not None and y_train is not None:
        if n_initial_samples > 0 and n_initial_samples < n_train:
            # Plot initial samples in gray
            axes[1].scatter(train_x[:n_initial_samples], train_y[:n_initial_samples], 
                          c=RWTH_GRAY, s=40, alpha=0.3, edgecolors=RWTH_BLACK, linewidth=0.8, zorder=4)
            
            # Plot exploration samples with RWTH colormap
            scatter2 = axes[1].scatter(train_x[n_initial_samples:], train_y[n_initial_samples:], 
                                      c=time_colors, cmap=RWTH_CMAP, 
                                      s=40, alpha=0.8, edgecolors=RWTH_BLACK, linewidth=0.8, zorder=5)
        else:
            scatter2 = axes[1].scatter(train_x, train_y, c=time_colors, cmap=RWTH_CMAP, 
                                      s=40, alpha=0.8, edgecolors=RWTH_BLACK, linewidth=0.8, zorder=5)
    
    axes[1].set_xlabel(label_map.get(dim_names[x_dim], dim_names[x_dim]), fontsize=14)
    axes[1].set_ylabel(label_map.get(dim_names[y_dim], dim_names[y_dim]), fontsize=14)
    axes[1].set_title(f'GP Prediction: {error_label_map.get(error_names[state_dim], error_names[state_dim])}', 
                     fontsize=15, fontweight='bold', pad=10)
    axes[1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Plot 3: GP uncertainty
    im3 = axes[2].contourf(X, Y, Z_std, levels=20, cmap='viridis')
    axes[2].contour(X, Y, Z_std, levels=10, colors='black', linewidths=0.3, alpha=0.3)
    
    if x_train is not None and y_train is not None:
        if n_initial_samples > 0 and n_initial_samples < n_train:
            # Plot initial samples in gray
            axes[2].scatter(train_x[:n_initial_samples], train_y[:n_initial_samples], 
                          c=RWTH_GRAY, s=40, alpha=0.3, edgecolors=RWTH_BLACK, linewidth=0.8, zorder=4)
            
            # Plot exploration samples with RWTH colormap
            scatter3 = axes[2].scatter(train_x[n_initial_samples:], train_y[n_initial_samples:], 
                                      c=time_colors, cmap=RWTH_CMAP, 
                                      s=40, alpha=0.8, edgecolors=RWTH_BLACK, linewidth=0.8, zorder=5)
            
            # Add colorbar for exploration samples
            cbar_samples = plt.colorbar(scatter3, ax=axes[2], pad=0.12, aspect=20)
            cbar_samples.set_label('Exploration step', fontsize=12, rotation=270, labelpad=20)
            cbar_samples.ax.tick_params(labelsize=10)
        else:
            scatter3 = axes[2].scatter(train_x, train_y, c=time_colors, cmap=RWTH_CMAP, 
                                      s=40, alpha=0.8, edgecolors=RWTH_BLACK, linewidth=0.8, zorder=5)
            
            # Add colorbar for sample order
            cbar_samples = plt.colorbar(scatter3, ax=axes[2], pad=0.12, aspect=20)
            cbar_samples.set_label('Sample order', fontsize=12, rotation=270, labelpad=20)
            cbar_samples.ax.tick_params(labelsize=10)
    
    axes[2].set_xlabel(label_map.get(dim_names[x_dim], dim_names[x_dim]), fontsize=14)
    axes[2].set_ylabel(label_map.get(dim_names[y_dim], dim_names[y_dim]), fontsize=14)
    axes[2].set_title(r'GP Uncertainty ($\sigma$): ' + error_label_map.get(error_names[state_dim], error_names[state_dim]), 
                     fontsize=15, fontweight='bold', pad=10)
    axes[2].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Add colorbars after tight_layout
    plt.tight_layout()
    
    # Shared colorbar for first two plots
    cbar1 = fig.colorbar(im2, ax=axes[:2], location='left', pad=0.08, aspect=30)
    cbar1.set_label('Model error', fontsize=12)
    cbar1.ax.tick_params(labelsize=10)
    
    # Colorbar for uncertainty plot
    cbar3 = fig.colorbar(im3, ax=axes[2], pad=0.02, aspect=30)
    cbar3.set_label('Std deviation', fontsize=12)
    cbar3.ax.tick_params(labelsize=10)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        # Also save as PDF
        # pdf_path = save_path.replace('.png', '.pdf')
        # plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {error_names[state_dim]} (2D) -> {save_path}")
        plt.close(fig)
    else:
        plt.show()
        print(f"  Displayed: {error_names[state_dim]} (2D)")



def print_statistics(true_error, gp_mean, gp_std, error_names):
    """Print model error statistics"""
    print("\n" + "="*60)
    print("Model Error Statistics")
    print("="*60)
    
    for dim in range(true_error.shape[1]):
        mse = np.mean((true_error[:, dim] - gp_mean[:, dim])**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_error[:, dim] - gp_mean[:, dim]))
        max_error = np.max(np.abs(true_error[:, dim] - gp_mean[:, dim]))
        
        # Coverage: percentage within 2σ
        residuals = np.abs(true_error[:, dim] - gp_mean[:, dim])
        coverage_2sigma = np.mean(residuals <= 2*gp_std[:, dim]) * 100
        coverage_1sigma = np.mean(residuals <= 1*gp_std[:, dim]) * 100
        
        # Mean uncertainty
        mean_std = np.mean(gp_std[:, dim])
        
        print(f"\n{error_names[dim]}:")
        print(f"  RMSE:          {rmse:.6f}")
        print(f"  MAE:           {mae:.6f}")
        print(f"  Max Error:     {max_error:.6f}")
        print(f"  Mean σ:        {mean_std:.6f}")
        print(f"  68% Coverage:  {coverage_1sigma:.1f}% (within 1σ)")
        print(f"  95% Coverage:  {coverage_2sigma:.1f}% (within 2σ)")


def plot_training_error_scatter(states, actions, true_error, gp_mean, dim_names, error_names, save_path=None, n_initial_samples=0):
    """Render a 3D scatter of training inputs colored by sample timing.

    Each point corresponds to a training tuple (dθ, θ, u). Initial samples are shown
    in gray, while exploration samples are colored by the RWTH colormap to show
    temporal progression.
    """
    if states.shape[1] != 2 or actions.shape[1] != 1:
        warnings.warn("3D scatter currently implemented for 2D state / 1D action setups.")
        return

    dtheta = states[:, 0]
    theta = states[:, 1]
    u = actions[:, 0]
    n_train = len(dtheta)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate initial samples from exploration samples
    if n_initial_samples > 0 and n_initial_samples < n_train:
        # Plot initial samples in gray
        ax.scatter(dtheta[:n_initial_samples], theta[:n_initial_samples], u[:n_initial_samples],
                  c=RWTH_GRAY, s=60, alpha=0.3, edgecolors=RWTH_BLACK, linewidth=0.5, 
                  depthshade=True, label='Initial samples')
        
        # Plot exploration samples with RWTH colormap
        n_exploration = n_train - n_initial_samples
        time_colors = np.arange(n_exploration)
        scatter = ax.scatter(dtheta[n_initial_samples:], theta[n_initial_samples:], u[n_initial_samples:],
                           c=time_colors, cmap=RWTH_CMAP, 
                           s=60, alpha=0.8, edgecolors=RWTH_BLACK, linewidth=0.5, 
                           depthshade=True, label='Exploration samples')
        
        # Add colorbar for exploration timing
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8, aspect=20)
        cbar.set_label('Exploration step', fontsize=12, rotation=270, labelpad=25)
        cbar.ax.tick_params(labelsize=10)
    else:
        # All samples with colormap (no distinction)
        time_colors = np.arange(n_train)
        scatter = ax.scatter(dtheta, theta, u, c=time_colors, cmap=RWTH_CMAP, 
                           s=60, alpha=0.8, edgecolors=RWTH_BLACK, linewidth=0.5, 
                           depthshade=True, label='Training data')
        
        # Add colorbar for sample timing
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8, aspect=20)
        cbar.set_label('Sample order', fontsize=12, rotation=270, labelpad=25)
        cbar.ax.tick_params(labelsize=10)

    # Improved axis labels
    label_map = {'dθ': r'$\dot{\vartheta}$ [rad/s]', 'θ': r'$\vartheta$ [rad]', 'u': r'$u$ [Nm]'}
    ax.set_xlabel(label_map.get(dim_names[0], dim_names[0]), fontsize=14, labelpad=10)
    ax.set_ylabel(label_map.get(dim_names[1], dim_names[1]), fontsize=14, labelpad=10)
    ax.set_zlabel(label_map.get(dim_names[2], dim_names[2]), fontsize=14, labelpad=10)
    ax.set_title('Training Data: Sample Acquisition Timeline', fontweight='bold', fontsize=16, pad=20)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        # Also save as PDF
        # pdf_path = save_path.replace('.png', '.pdf')
        # plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"  Saved: training error mismatch (3D scatter) -> {save_path}")
        
        # Also save as pickle for interactive viewing
        pkl_path = save_path.replace('.png', '.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(fig, f)
        print(f"  Saved: interactive figure (pickle) -> {pkl_path}")
        
        plt.close(fig)
    else:
        plt.show()
        print("  Displayed: training error mismatch (3D scatter)")
