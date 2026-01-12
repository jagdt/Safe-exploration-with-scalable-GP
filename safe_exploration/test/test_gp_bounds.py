#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for comparing GP confidence bounds between NumpyGPModel and ScalableGPModel.

Tests both models on various synthetic functions to evaluate:
- Beta values (confidence bound scaling)
- Projection errors (for ScalableGP)
- Coverage (how well bounds capture true function)
- Bound tightness

Usage:
    python test_gp_bounds.py [--functions sin,quadratic] [--dims 1,2] [--n-train 50]
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from safe_exploration.ssm_numpy.gp_numpy import NumpyGPModel
from safe_exploration.ssm_numpy.scalable_gp import ScalableGPModel


# =============================================================================
# Test Functions (reuse from test_gp_fitting.py)
# =============================================================================

def linear_1d(x):
    """Linear function: f(x) = 2x + 1"""
    return 2 * x[:, 0] + 1

def quadratic_1d(x):
    """Quadratic function: f(x) = x²"""
    return x[:, 0] ** 2

def sin_1d(x):
    """Sinusoidal function: f(x) = sin(2πx)"""
    return np.sin(2 * np.pi * x[:, 0])

def exp_decay_1d(x):
    """Exponential decay: f(x) = exp(-x²)"""
    return np.exp(-x[:, 0] ** 2)

def linear_2d(x):
    """Linear function: f(x) = x₁ + 2x₂"""
    return x[:, 0] + 2 * x[:, 1]

def quadratic_2d(x):
    """Quadratic function: f(x) = x₁² + x₂²"""
    return x[:, 0] ** 2 + x[:, 1] ** 2

def sin_2d(x):
    """Sinusoidal function: f(x) = sin(2πx₁) * cos(2πx₂)"""
    return np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])

# Function registry
TEST_FUNCTIONS = {
    1: {
        'linear': linear_1d,
        'quadratic': quadratic_1d,
        'sin': sin_1d,
        'exp_decay': exp_decay_1d,
    },
    2: {
        'linear': linear_2d,
        'quadratic': quadratic_2d,
        'sin': sin_2d,
    },
}


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(func, n_train, n_test, dim, domain=(-1, 1), noise_std=0.0, seed=42):
    """Generate training and test data for a given function."""
    np.random.seed(seed)
    
    # Training data (random)
    X_train = np.random.uniform(domain[0], domain[1], (n_train, dim))
    y_train_true = func(X_train)
    y_train = y_train_true + noise_std * np.random.randn(n_train)
    
    # Test data (grid for 1D/2D, random for 3D+)
    if dim == 1:
        X_test = np.linspace(domain[0], domain[1], n_test).reshape(-1, 1)
    elif dim == 2:
        n_per_dim = int(np.sqrt(n_test))
        x1 = np.linspace(domain[0], domain[1], n_per_dim)
        x2 = np.linspace(domain[0], domain[1], n_per_dim)
        X1, X2 = np.meshgrid(x1, x2)
        X_test = np.column_stack([X1.ravel(), X2.ravel()])
    else:
        X_test = np.random.uniform(domain[0], domain[1], (n_test, dim))
    
    y_test_true = func(X_test)
    
    return X_train, y_train.reshape(-1, 1), X_test, y_test_true.reshape(-1, 1)


# =============================================================================
# Bounds Evaluation
# =============================================================================

def compute_bound_metrics(y_true, y_pred, y_std, beta, projection_error=0.0):
    """Compute metrics for GP confidence bounds.
    
    Parameters
    ----------
    y_true : ndarray [N × 1]
        True values
    y_pred : ndarray [N × 1]
        Predicted mean
    y_std : ndarray [N × 1]
        Predicted standard deviation
    beta : float or ndarray [1]
        Confidence bound scaling factor
    projection_error : float or ndarray [1]
        Projection error offset (for scalable GP)
        
    Returns
    -------
    metrics : dict
        Dictionary of metrics
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    y_std = y_std.ravel()
    
    if isinstance(beta, np.ndarray):
        beta = beta[0]
    if isinstance(projection_error, np.ndarray):
        projection_error = projection_error[0]
    
    # Compute upper and lower confidence bounds
    # UCB/LCB = μ ± √β * σ + ε_proj
    conf_width = np.sqrt(beta) * y_std + projection_error
    upper_bound = y_pred + conf_width
    lower_bound = y_pred - conf_width
    
    # Coverage: fraction of test points within bounds
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    
    # Average bound width
    avg_width = np.mean(2 * conf_width)
    
    # Max violation (how far outside bounds)
    violations_upper = np.maximum(0, y_true - upper_bound)
    violations_lower = np.maximum(0, lower_bound - y_true)
    max_violation = np.max(violations_upper + violations_lower)
    avg_violation = np.mean(violations_upper + violations_lower)
    
    # Tightness: average slack when predictions are correct
    slack_upper = np.where(y_true <= upper_bound, upper_bound - y_true, 0)
    slack_lower = np.where(y_true >= lower_bound, y_true - lower_bound, 0)
    avg_slack = np.mean(slack_upper + slack_lower)
    
    # Normalized coverage efficiency: coverage / average_width
    # Higher is better (high coverage with tight bounds)
    efficiency = coverage / avg_width if avg_width > 0 else 0
    
    return {
        'beta': beta,
        'projection_error': projection_error,
        'coverage': coverage,
        'avg_width': avg_width,
        'max_violation': max_violation,
        'avg_violation': avg_violation,
        'avg_slack': avg_slack,
        'efficiency': efficiency,
    }


def fit_and_evaluate_bounds(gp_class, X_train, y_train, X_test, y_test_true,
                            kern_type='rbf', opt_hyp=True, delta=0.05,
                            rkhs_norm=1.0, R_subgaussian=1.0, **gp_kwargs):
    """Fit a GP, compute bounds, and evaluate on test data.
    
    Parameters
    ----------
    gp_class : class
        GP model class (NumpyGPModel or ScalableGPModel)
    X_train, y_train : ndarray
        Training data
    X_test, y_test_true : ndarray
        Test data
    kern_type : str
        Kernel type
    opt_hyp : bool
        Whether to optimize hyperparameters
    delta : float
        Confidence parameter (1-δ confidence)
    rkhs_norm : float
        RKHS norm bound
    R_subgaussian : float
        Sub-Gaussian noise parameter
    **gp_kwargs : dict
        Additional arguments for GP constructor
        
    Returns
    -------
    bound_metrics : dict
        Bound evaluation metrics
    y_pred, y_std : ndarray
        Predictions
    gp : GP model
        Trained GP model
    """
    n_train, dim = X_train.shape
    
    # Create GP
    gp = gp_class(
        n_s_out=1,
        n_s_in=dim,
        n_u=0,
        kern_types=[kern_type],
        **gp_kwargs
    )
    
    # Train
    t_start = time()
    gp.train(X_train, y_train, opt_hyp=opt_hyp)
    train_time = time() - t_start
    
    # Compute bounds
    t_start = time()
    if gp_class.__name__ == 'ScalableGPModel':
        gp.compute_bounds(delta=delta, rkhs_norm=rkhs_norm, 
                         R_subgaussian=R_subgaussian, projection_error=None)
        projection_error = gp.projection_error_per_dim[0] if gp.projection_error_per_dim is not None else 0.0
    else:
        gp.compute_bounds(delta=delta, rkhs_norm=rkhs_norm, 
                         R_subgaussian=R_subgaussian)
        projection_error = 0.0
    
    beta = gp.beta_safety_per_dim[0]
    bounds_time = time() - t_start
    
    # Predict
    t_start = time()
    y_pred, y_std = gp.predict(X_test)
    predict_time = time() - t_start
    
    # Compute bound metrics
    bound_metrics = compute_bound_metrics(y_test_true, y_pred, y_std, beta, projection_error)
    bound_metrics['train_time'] = train_time
    bound_metrics['bounds_time'] = bounds_time
    bound_metrics['predict_time'] = predict_time
    
    return bound_metrics, y_pred, y_std, gp


# =============================================================================
# Visualization
# =============================================================================

def plot_1d_bounds_comparison(X_train, y_train, X_test, y_test_true,
                              results, func_name, save_path=None):
    """Plot 1D bounds comparison between GP models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for ax, (model_name, res) in zip(axes, results.items()):
        y_pred = res['y_pred'].ravel()
        y_std = res['y_std'].ravel()
        x_test = X_test.ravel()
        
        beta = res['metrics']['beta']
        proj_err = res['metrics']['projection_error']
        conf_width = np.sqrt(beta) * y_std + proj_err
        
        # Sort for plotting
        sort_idx = np.argsort(x_test)
        x_test = x_test[sort_idx]
        y_pred = y_pred[sort_idx]
        conf_width = conf_width[sort_idx]
        y_true = y_test_true.ravel()[sort_idx]
        
        # Plot confidence bounds
        ax.fill_between(x_test, y_pred - conf_width, y_pred + conf_width,
                       alpha=0.3, label=f'±√β·σ + ε (β={beta:.2f})', color='blue')
        ax.plot(x_test, y_pred, 'b-', label='Prediction', linewidth=2)
        ax.plot(x_test, y_true, 'k--', label='True', linewidth=1.5, alpha=0.8)
        ax.scatter(X_train.ravel(), y_train.ravel(), c='r', s=30,
                  zorder=5, label='Training data')
        
        # Mark violations
        upper_bound = y_pred + conf_width
        lower_bound = y_pred - conf_width
        violations = (y_true > upper_bound) | (y_true < lower_bound)
        if np.any(violations):
            ax.scatter(x_test[violations], y_true[violations], c='orange', s=80,
                      marker='o', facecolors='none', edgecolors='orange', linewidths=2,
                      label='Violations', zorder=6)
        
        m = res['metrics']
        title = (f'{model_name}\n'
                f'Coverage: {m["coverage"]:.1%}, Avg Width: {m["avg_width"]:.3f}\n'
                f'Max Violation: {m["max_violation"]:.3f}')
        if proj_err > 0:
            title += f'\nProj Error: {proj_err:.4f}'
        
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Confidence Bounds: {func_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_2d_bounds_comparison(X_train, y_train, X_test, y_test_true,
                              results, func_name, save_path=None):
    """Plot 2D bounds comparison between GP models."""
    n_per_dim = int(np.sqrt(len(X_test)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reshape for contour plots
    X1 = X_test[:, 0].reshape(n_per_dim, n_per_dim)
    X2 = X_test[:, 1].reshape(n_per_dim, n_per_dim)
    Y_true = y_test_true.reshape(n_per_dim, n_per_dim)
    
    # True function
    ax = axes[0, 0]
    c = ax.contourf(X1, X2, Y_true, levels=20, cmap='viridis')
    ax.scatter(X_train[:, 0], X_train[:, 1], c='red', s=20, marker='x')
    ax.set_title('True Function')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    plt.colorbar(c, ax=ax)
    
    for col, (model_name, res) in enumerate(results.items(), 1):
        y_pred = res['y_pred'].ravel()
        y_std = res['y_std'].ravel()
        
        beta = res['metrics']['beta']
        proj_err = res['metrics']['projection_error']
        conf_width = np.sqrt(beta) * y_std + proj_err
        
        # Upper bound
        ax = axes[0, col]
        upper_bound = (y_pred + conf_width).reshape(n_per_dim, n_per_dim)
        c = ax.contourf(X1, X2, upper_bound, levels=20, cmap='viridis')
        ax.scatter(X_train[:, 0], X_train[:, 1], c='red', s=20, marker='x')
        ax.set_title(f'{model_name} Upper Bound\nβ={beta:.2f}')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        plt.colorbar(c, ax=ax)
        
        # Lower bound
        ax = axes[1, col]
        lower_bound = (y_pred - conf_width).reshape(n_per_dim, n_per_dim)
        c = ax.contourf(X1, X2, lower_bound, levels=20, cmap='viridis')
        ax.scatter(X_train[:, 0], X_train[:, 1], c='red', s=20, marker='x')
        m = res['metrics']
        ax.set_title(f'{model_name} Lower Bound\n'
                    f'Coverage: {m["coverage"]:.1%}')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        plt.colorbar(c, ax=ax)
    
    # Coverage visualization
    ax = axes[1, 0]
    ax.axis('off')
    
    # Add text summary
    summary_text = "Bound Metrics:\n\n"
    for model_name, res in results.items():
        m = res['metrics']
        summary_text += f"{model_name}:\n"
        summary_text += f"  Coverage: {m['coverage']:.1%}\n"
        summary_text += f"  Avg Width: {m['avg_width']:.3f}\n"
        summary_text += f"  Max Violation: {m['max_violation']:.3f}\n"
        if m['projection_error'] > 0:
            summary_text += f"  Proj Error: {m['projection_error']:.4f}\n"
        summary_text += "\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
           verticalalignment='center', transform=ax.transAxes)
    
    fig.suptitle(f'Confidence Bounds: {func_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_bounds_table(all_results):
    """Print bounds results as a formatted table."""
    print("\n" + "=" * 120)
    print("GP BOUNDS COMPARISON RESULTS")
    print("=" * 120)
    
    header = (f"{'Function':<20} {'Model':<15} {'β':<10} {'Proj Err':<12} "
             f"{'Coverage':<12} {'Avg Width':<12} {'Max Viol':<12} {'Efficiency':<12}")
    print(header)
    print("-" * 120)
    
    for (dim, func_name), results in all_results.items():
        func_label = f"{dim}D-{func_name}"
        for model_name, res in results.items():
            m = res['metrics']
            proj_err_str = f"{m['projection_error']:.4f}" if m['projection_error'] > 1e-10 else "N/A"
            row = (f"{func_label:<20} {model_name:<15} {m['beta']:<10.2f} {proj_err_str:<12} "
                  f"{m['coverage']:<12.1%} {m['avg_width']:<12.4f} "
                  f"{m['max_violation']:<12.4f} {m['efficiency']:<12.4f}")
            print(row)
            func_label = ""  # Only print once per function
        print("-" * 120)


# =============================================================================
# Main Test Runner
# =============================================================================

def run_bounds_tests(functions=None, dims=None, n_train=50, n_test=200,
                    noise_std=0.01, kern_type='rbf', opt_hyp=True,
                    n_frequencies=15, period=3.0, delta=0.05,
                    rkhs_norm=1.0, R_subgaussian=1.0,
                    visualize=True, save_dir=None):
    """Run GP bounds comparison tests.
    
    Parameters
    ----------
    functions : list of str, optional
        Function names to test (default: all)
    dims : list of int, optional
        Dimensions to test (default: [1, 2])
    n_train : int
        Number of training points
    n_test : int
        Number of test points
    noise_std : float
        Observation noise
    kern_type : str
        Kernel type
    opt_hyp : bool
        Whether to optimize hyperparameters
    n_frequencies : int
        Number of frequencies for ScalableGP
    period : float
        Period for ScalableGP
    delta : float
        Confidence parameter (1-δ confidence)
    rkhs_norm : float
        RKHS norm bound
    R_subgaussian : float
        Sub-Gaussian noise parameter
    visualize : bool
        Whether to create plots
    save_dir : str, optional
        Directory to save plots
        
    Returns
    -------
    all_results : dict
        Results for all tests
    """
    if dims is None:
        dims = [1, 2]
    
    all_results = {}
    
    print(f"\n{'='*80}")
    print(f"GP BOUNDS TEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Kernel type: {kern_type}")
    print(f"  Training points: {n_train}")
    print(f"  Test points: {n_test}")
    print(f"  Noise std: {noise_std}")
    print(f"  Confidence δ: {delta} (implies {100*(1-delta):.1f}% confidence)")
    print(f"  RKHS norm bound: {rkhs_norm}")
    print(f"  R (sub-Gaussian): {R_subgaussian}")
    print(f"  ScalableGP frequencies: {n_frequencies}")
    print(f"  ScalableGP period: {period}")
    print(f"{'='*80}\n")
    
    for dim in dims:
        if dim not in TEST_FUNCTIONS:
            print(f"Skipping dimension {dim} (not supported)")
            continue
        
        available_funcs = TEST_FUNCTIONS[dim]
        funcs_to_test = functions if functions else list(available_funcs.keys())
        
        for func_name in funcs_to_test:
            if func_name not in available_funcs:
                print(f"Skipping {func_name} for {dim}D (not available)")
                continue
            
            print(f"\n{'='*60}")
            print(f"Testing {dim}D {func_name} function")
            print(f"{'='*60}")
            
            func = available_funcs[func_name]
            
            # Generate data
            X_train, y_train, X_test, y_test_true = generate_data(
                func, n_train, n_test, dim, noise_std=noise_std
            )
            
            results = {}
            
            # Test NumpyGPModel
            print(f"\n--- NumpyGPModel ---")
            try:
                metrics, y_pred, y_std, gp = fit_and_evaluate_bounds(
                    NumpyGPModel, X_train, y_train, X_test, y_test_true,
                    kern_type=kern_type, opt_hyp=opt_hyp,
                    delta=delta, rkhs_norm=rkhs_norm, R_subgaussian=R_subgaussian
                )
                results['NumpyGP'] = {
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_std': y_std,
                    'gp': gp,
                }
                print(f"  β: {metrics['beta']:.4f}")
                print(f"  Coverage: {metrics['coverage']:.1%}")
                print(f"  Avg bound width: {metrics['avg_width']:.4f}")
                print(f"  Max violation: {metrics['max_violation']:.4f}")
                print(f"  Train time: {metrics['train_time']:.3f}s")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
            
            # Test ScalableGPModel
            print(f"\n--- ScalableGPModel ---")
            try:
                metrics, y_pred, y_std, gp = fit_and_evaluate_bounds(
                    ScalableGPModel, X_train, y_train, X_test, y_test_true,
                    kern_type=kern_type, opt_hyp=opt_hyp,
                    n_frequencies=n_frequencies, periods=period,
                    delta=delta, rkhs_norm=rkhs_norm, R_subgaussian=R_subgaussian
                )
                results['ScalableGP'] = {
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_std': y_std,
                    'gp': gp,
                }
                print(f"  β: {metrics['beta']:.4f}")
                print(f"  Projection error: {metrics['projection_error']:.6f}")
                print(f"  Coverage: {metrics['coverage']:.1%}")
                print(f"  Avg bound width: {metrics['avg_width']:.4f}")
                print(f"  Max violation: {metrics['max_violation']:.4f}")
                print(f"  Train time: {metrics['train_time']:.3f}s")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
            
            all_results[(dim, func_name)] = results
            
            # Visualize
            if visualize and len(results) > 0:
                save_path = None
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f'{dim}D_{func_name}_bounds.png')
                
                if dim == 1:
                    plot_1d_bounds_comparison(X_train, y_train, X_test, y_test_true,
                                            results, f'{dim}D {func_name}', save_path)
                elif dim == 2:
                    plot_2d_bounds_comparison(X_train, y_train, X_test, y_test_true,
                                            results, f'{dim}D {func_name}', save_path)
    
    # Print summary table
    print_bounds_table(all_results)
    
    return all_results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test GP confidence bounds')
    parser.add_argument('--functions', type=str, default=None,
                       help='Comma-separated list of functions to test')
    parser.add_argument('--dims', type=str, default='1,2',
                       help='Comma-separated list of dimensions to test')
    parser.add_argument('--n-train', type=int, default=50,
                       help='Number of training points')
    parser.add_argument('--n-test', type=int, default=200,
                       help='Number of test points')
    parser.add_argument('--noise', type=float, default=0.01,
                       help='Observation noise std')
    parser.add_argument('--kernel', type=str, default='rbf',
                       help='Kernel type (rbf, mat52, sum_lin_rbf)')
    parser.add_argument('--no-opt', action='store_true',
                       help='Disable hyperparameter optimization')
    parser.add_argument('--n-freq', type=int, default=15,
                       help='Number of frequencies for ScalableGP')
    parser.add_argument('--period', type=float, default=3.0,
                       help='Period for ScalableGP')
    parser.add_argument('--delta', type=float, default=0.05,
                       help='Confidence parameter (1-delta confidence)')
    parser.add_argument('--rkhs-norm', type=float, default=1.0,
                       help='RKHS norm bound')
    parser.add_argument('--R', type=float, default=1.0,
                       help='Sub-Gaussian noise parameter')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    functions = args.functions.split(',') if args.functions else None
    dims = [int(d) for d in args.dims.split(',')]
    
    run_bounds_tests(
        functions=functions,
        dims=dims,
        n_train=args.n_train,
        n_test=args.n_test,
        noise_std=args.noise,
        kern_type=args.kernel,
        opt_hyp=not args.no_opt,
        n_frequencies=args.n_freq,
        period=args.period,
        delta=args.delta,
        rkhs_norm=args.rkhs_norm,
        R_subgaussian=args.R,
        visualize=not args.no_viz,
        save_dir=args.save_dir,
    )
