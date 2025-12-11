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
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os


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


def plot_model_error_comparison(safempc, env, save_dir=None, n_points=30, plot_bounds=False):
    """Create comprehensive model error comparison plots
    
    Parameters
    ----------
    safempc : SimpleSafeMPC
        The MPC controller with trained GP
    save_dir : str, optional
        Directory to save plots
    n_points : int
        Number of points per dimension for grid
    """
    
    if save_dir is None:
        save_dir = './model_error_plots'
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
        test_inputs_2d_theta_u = np.hstack([states_2d_theta_u_trafo, actions_2d_theta_u])
        test_inputs_2d_dtheta_u = np.hstack([states_2d_dtheta_u_trafo, actions_2d_dtheta_u])
    else:
        test_inputs_1d_u = np.hstack([states_1d_u, actions_1d_u])
        test_inputs_1d_theta = np.hstack([states_1d_theta, actions_1d_theta])
        test_inputs_1d_dtheta = np.hstack([states_1d_dtheta, actions_1d_dtheta])
    
        test_inputs_2d_theta_u = np.hstack([states_2d_theta_u, actions_2d_theta_u])
        test_inputs_2d_dtheta_u = np.hstack([states_2d_dtheta_u, actions_2d_dtheta_u])
    
    gp_mean_1d_u, gp_std_1d_u = safempc.ssm.predict(test_inputs_1d_u)
    gp_mean_1d_theta, gp_std_1d_theta = safempc.ssm.predict(test_inputs_1d_theta)
    gp_mean_1d_dtheta, gp_std_1d_dtheta = safempc.ssm.predict(test_inputs_1d_dtheta)

    gp_mean_2d_theta_u, gp_std_2d_theta_u = safempc.ssm.predict(test_inputs_2d_theta_u)
    gp_mean_2d_dtheta_u, gp_std_2d_dtheta_u = safempc.ssm.predict(test_inputs_2d_dtheta_u)
    
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
    if hasattr(safempc.ssm, 'projection_error'):
        proj_error = safempc.ssm.projection_error
    elif hasattr(safempc, 'projection_error'):
        proj_error = safempc.projection_error
    
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

    # Plot 1D slice varying u
    for dim in range(safempc.n_s):
        beta_dim = get_param_for_dim(beta, dim, safempc.n_s)
        proj_dim = get_param_for_dim(proj_error, dim, safempc.n_s)
        
        plot_1d_comparison(
            states_1d_u, actions_1d_u, true_error_1d_u, gp_mean_1d_u, gp_std_1d_u,
            state_dim=dim, vary_dim=2,  # u is at index 2
            dim_names=dim_names, error_names=error_names,
            save_path=os.path.join(save_dir, f'model_error_1d_u_dim{dim}.png'),
            x_train=x_train, y_train=y_train,
            beta=beta_dim, proj_error=proj_dim
        )
    
    # Plot 1D slice varying theta
    for dim in range(safempc.n_s):
        beta_dim = get_param_for_dim(beta, dim, safempc.n_s)
        proj_dim = get_param_for_dim(proj_error, dim, safempc.n_s)
        
        plot_1d_comparison(
            states_1d_theta, actions_1d_theta, true_error_1d_theta, gp_mean_1d_theta, gp_std_1d_theta,
            state_dim=dim, vary_dim=1,  # theta is at index 1
            dim_names=dim_names, error_names=error_names,
            save_path=os.path.join(save_dir, f'model_error_1d_theta_dim{dim}.png'),
            x_train=x_train, y_train=y_train,
            beta=beta_dim, proj_error=proj_dim
        )
    
    # Plot 1D slice varying dtheta
    for dim in range(safempc.n_s):
        beta_dim = get_param_for_dim(beta, dim, safempc.n_s)
        proj_dim = get_param_for_dim(proj_error, dim, safempc.n_s)
        
        plot_1d_comparison(
            states_1d_dtheta, actions_1d_dtheta, true_error_1d_dtheta, gp_mean_1d_dtheta, gp_std_1d_dtheta,
            state_dim=dim, vary_dim=0,  # dtheta is at index 0
            dim_names=dim_names, error_names=error_names,
            save_path=os.path.join(save_dir, f'model_error_1d_dtheta_dim{dim}.png'),
            x_train=x_train, y_train=y_train,
            beta=beta_dim, proj_error=proj_dim
        )
    
    # Plot theta vs u
    for dim in range(safempc.n_s):
        plot_2d_comparison(
            states_2d_theta_u, actions_2d_theta_u, true_error_2d_theta_u, 
            gp_mean_2d_theta_u, gp_std_2d_theta_u,
            state_dim=dim, 
            vary_dims=(1, safempc.n_s),
            n_points=n_points,
            dim_names=dim_names, error_names=error_names,
            save_path=os.path.join(save_dir, f'model_error_2d_theta_u_dim{dim}.png'),
            x_train=x_train, y_train=y_train
        )
    
    # Plot dtheta vs u
    for dim in range(safempc.n_s):
        plot_2d_comparison(
            states_2d_dtheta_u, actions_2d_dtheta_u, true_error_2d_dtheta_u, 
            gp_mean_2d_dtheta_u, gp_std_2d_dtheta_u,
            state_dim=dim, 
            vary_dims=(0, safempc.n_s),
            n_points=n_points,
            dim_names=dim_names, error_names=error_names,
            save_path=os.path.join(save_dir, f'model_error_2d_dtheta_u_dim{dim}.png'),
            x_train=x_train, y_train=y_train
        )

    # 3D scatter of training data highlighting residual mismatch
    gp_mean_train, _ = safempc.ssm.predict(x_train_gp)
    true_error_train = compute_true_model_error(safempc, env, states_train_orig, actions_train)
    plot_training_error_scatter(
        states_train_orig,
        actions_train,
        true_error_train,
        gp_mean_train,
        dim_names,
        error_names
    )
    
    print_statistics(true_error_2d_theta_u, gp_mean_2d_theta_u, gp_std_2d_theta_u, error_names)
    
    print(f"\n{'='*60}")
    print(f"All plots displayed")
    print(f"{'='*60}\n")


def plot_1d_comparison(states, actions, true_error, gp_mean, gp_std,
                       state_dim, vary_dim, dim_names, error_names, save_path, x_train=None, y_train=None,
                       beta=2.0, proj_error=0.0):
    """Plot 1D comparison of true vs predicted model error"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
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
    ax.plot(x_sorted, true_sorted, 'b-', linewidth=2.5, label='True model error', alpha=0.8)
    ax.plot(x_sorted, gp_sorted, 'r--', linewidth=2, label='GP prediction', alpha=0.8)
    ax.fill_between(x_sorted,
                    gp_sorted - 2*gp_std_sorted,
                    gp_sorted + 2*gp_std_sorted,
                    color='red', alpha=0.1, label='GP ±2σ')
    
    # Plot Confidence Bound (Safety Bound)
    plot_bound = True
    if plot_bound:
        bound = beta * gp_std_sorted + proj_error
        ax.plot(x_sorted, gp_sorted + bound, 'k--', linewidth=1.5, label='Safety Bound', alpha=0.7)
        ax.plot(x_sorted, gp_sorted - bound, 'k--', linewidth=1.5, alpha=0.7)
        ax.fill_between(x_sorted,
                        gp_sorted - bound,
                    gp_sorted + bound,
                    color='gray', alpha=0.2, label='Confidence Region')
    
    # Plot training points if provided
    if x_train is not None and y_train is not None:
        train_x = x_train[:, vary_dim]
        train_y = y_train[:, state_dim]
        ax.scatter(train_x, train_y, c='green', s=50, alpha=0.6, 
                  edgecolors='darkgreen', linewidth=1, label='Training data', zorder=5)
    
    ax.axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel(f'{dim_names[vary_dim]}', fontsize=12)
    ax.set_ylabel(f'Model error: {error_names[state_dim]}', fontsize=12)
    ax.set_title(f'True vs GP Model Error', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print(f"  Displayed: {error_names[state_dim]} (1D)")


def plot_2d_comparison(states, actions, true_error, gp_mean, gp_std,
                       state_dim, vary_dims, n_points, dim_names, error_names, save_path, x_train=None, y_train=None):
    """Plot 2D heatmaps comparing true vs predicted model error"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    
    # Plot 1: True model error
    im1 = axes[0].contourf(X, Y, Z_true, levels=levels, cmap='RdBu_r', norm=norm)
    if x_train is not None and y_train is not None:
        train_x = x_train[:, x_dim]
        train_y = x_train[:, y_dim]
        axes[0].scatter(train_x, train_y, c='green', s=30, alpha=0.7, 
                       edgecolors='darkgreen', linewidth=0.5, label='Training data', zorder=5)
    axes[0].set_xlabel(dim_names[x_dim], fontsize=12)
    axes[0].set_ylabel(dim_names[y_dim], fontsize=12)
    axes[0].set_title(f'True Error: {error_names[state_dim]}', fontsize=13, fontweight='bold')
    
    # Plot 2: GP predicted error
    im2 = axes[1].contourf(X, Y, Z_gp, levels=levels, cmap='RdBu_r', norm=norm)
    if x_train is not None and y_train is not None:
        train_x = x_train[:, x_dim]
        train_y = x_train[:, y_dim]
        axes[1].scatter(train_x, train_y, c='green', s=30, alpha=0.7, 
                       edgecolors='darkgreen', linewidth=0.5, label='Training data', zorder=5)
    axes[1].set_xlabel(dim_names[x_dim], fontsize=12)
    axes[1].set_ylabel(dim_names[y_dim], fontsize=12)
    axes[1].set_title(f'GP Predicted: {error_names[state_dim]}', fontsize=13, fontweight='bold')
    
    # Plot 3: GP uncertainty
    im3 = axes[2].contourf(X, Y, Z_std, levels=20, cmap='viridis')
    if x_train is not None and y_train is not None:
        train_x = x_train[:, x_dim]
        train_y = x_train[:, y_dim]
        axes[2].scatter(train_x, train_y, c='green', s=30, alpha=0.7, 
                       edgecolors='darkgreen', linewidth=0.5, label='Training data', zorder=5)
    axes[2].set_xlabel(dim_names[x_dim], fontsize=12)
    axes[2].set_ylabel(dim_names[y_dim], fontsize=12)
    axes[2].set_title(f'GP Uncertainty (σ): {error_names[state_dim]}', fontsize=13, fontweight='bold')
    
    # Add colorbars after tight_layout
    plt.tight_layout()
    
    # Shared colorbar for first two plots on the left side (use im1 or im2, they have same scale)
    cbar1 = fig.colorbar(im2, ax=axes[:2], location='left', pad=0.15)
    cbar1.set_label('Error', fontsize=10)
    
    # Colorbar for uncertainty plot
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label('Std Dev', fontsize=10)
    
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


def plot_training_error_scatter(states, actions, true_error, gp_mean, dim_names, error_names):
    """Render a 3D scatter of training inputs colored by model mismatch.

    Each point corresponds to a training tuple (dθ, θ, u). Color encodes the
    norm of `true_error - gp_mean`, so vivid hues highlight where the GP
    deviates most from the ground-truth residuals.
    """

    if states.shape[1] != 2 or actions.shape[1] != 1:
        warnings.warn("3D scatter currently implemented for 2D state / 1D action setups.")
        return

    dtheta = states[:, 0]
    theta = states[:, 1]
    u = actions[:, 0]

    residual_mismatch = np.linalg.norm(true_error - gp_mean, axis=1)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(dtheta, theta, u, c=residual_mismatch, cmap='viridis', s=35, depthshade=True)

    ax.set_xlabel(dim_names[0])
    ax.set_ylabel(dim_names[1])
    ax.set_zlabel(dim_names[2])
    ax.set_title('Training Samples: |True Error − GP|', fontweight='bold')

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Residual mismatch (norm)', fontsize=10)

    plt.tight_layout()
    plt.show()
    print("  Displayed: training error mismatch (3D scatter)")
