#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for comparing GP fitting quality between NumpyGPModel and ScalableGPModel.

Tests both models on various synthetic functions in 1D, 2D, and 3D to evaluate:
- Fitting accuracy (RMSE, R²)
- Uncertainty calibration
- Computational time

Usage:
    python test_gp_fitting.py [--functions linear,quadratic,sin] [--dims 1,2] [--n-train 50]
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
# Test Functions
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

def step_1d(x):
    """Step function: f(x) = sign(x)"""
    return np.sign(x[:, 0])

def linear_2d(x):
    """Linear function: f(x) = x₁ + 2x₂"""
    return x[:, 0] + 2 * x[:, 1]

def quadratic_2d(x):
    """Quadratic function: f(x) = x₁² + x₂²"""
    return x[:, 0] ** 2 + x[:, 1] ** 2

def sin_2d(x):
    """Sinusoidal function: f(x) = sin(2πx₁) * cos(2πx₂)"""
    return np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])

def rosenbrock_2d(x):
    """Rosenbrock function (scaled): f(x) = (1-x₁)² + 10(x₂-x₁²)²"""
    return (1 - x[:, 0]) ** 2 + 10 * (x[:, 1] - x[:, 0] ** 2) ** 2

def linear_3d(x):
    """Linear function: f(x) = x₁ + 2x₂ + 3x₃"""
    return x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2]

def quadratic_3d(x):
    """Quadratic function: f(x) = x₁² + x₂² + x₃²"""
    return x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2

def sin_3d(x):
    """Sinusoidal function: f(x) = sin(2π(x₁+x₂+x₃))"""
    return np.sin(2 * np.pi * (x[:, 0] + x[:, 1] + x[:, 2]))


# Function registry
TEST_FUNCTIONS = {
    1: {
        'linear': linear_1d,
        'quadratic': quadratic_1d,
        'sin': sin_1d,
        'exp_decay': exp_decay_1d,
        'step': step_1d,
    },
    2: {
        'linear': linear_2d,
        'quadratic': quadratic_2d,
        'sin': sin_2d,
        'rosenbrock': rosenbrock_2d,
    },
    3: {
        'linear': linear_3d,
        'quadratic': quadratic_3d,
        'sin': sin_3d,
    },
}


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(func, n_train, n_test, dim, domain=(-1, 1), noise_std=0.0, seed=42):
    """Generate training and test data for a given function.
    
    Parameters
    ----------
    func : callable
        Function to fit
    n_train : int
        Number of training points
    n_test : int
        Number of test points
    dim : int
        Input dimension
    domain : tuple
        (min, max) for each dimension
    noise_std : float
        Standard deviation of observation noise
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    X_train, y_train, X_test, y_test, y_test_true
    """
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
    y_test = y_test_true + noise_std * np.random.randn(len(X_test))
    
    return X_train, y_train.reshape(-1, 1), X_test, y_test.reshape(-1, 1), y_test_true.reshape(-1, 1)


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(y_true, y_pred, y_std):
    """Compute evaluation metrics.
    
    Parameters
    ----------
    y_true : ndarray [N × 1]
        True values
    y_pred : ndarray [N × 1]
        Predicted mean
    y_std : ndarray [N × 1]
        Predicted standard deviation
        
    Returns
    -------
    metrics : dict
        Dictionary of metrics
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    y_std = y_std.ravel()
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Negative log predictive density (NLPD)
    # Lower is better
    nlpd = 0.5 * np.mean(np.log(2 * np.pi * y_std ** 2) + ((y_true - y_pred) / y_std) ** 2)
    
    # Calibration: fraction of test points within 2σ
    within_2sigma = np.mean(np.abs(y_true - y_pred) <= 2 * y_std)
    
    # Mean predicted uncertainty
    mean_std = np.mean(y_std)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'nlpd': nlpd,
        'within_2sigma': within_2sigma,
        'mean_std': mean_std,
    }


# =============================================================================
# GP Fitting
# =============================================================================

def fit_and_evaluate(gp_class, X_train, y_train, X_test, y_test_true, 
                     kern_type='rbf', opt_hyp=True, **gp_kwargs):
    """Fit a GP and evaluate on test data.
    
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
    **gp_kwargs : dict
        Additional arguments for GP constructor
        
    Returns
    -------
    metrics : dict
        Evaluation metrics
    y_pred, y_std : ndarray
        Predictions
    train_time : float
        Training time in seconds
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
    
    # Predict
    t_start = time()
    y_pred, y_std = gp.predict(X_test)
    predict_time = time() - t_start
    
    # Compute metrics
    metrics = compute_metrics(y_test_true, y_pred, y_std)
    metrics['train_time'] = train_time
    metrics['predict_time'] = predict_time
    
    return metrics, y_pred, y_std, gp


# =============================================================================
# Visualization
# =============================================================================

def plot_1d_comparison(X_train, y_train, X_test, y_test_true, 
                       results, func_name, save_path=None):
    """Plot 1D comparison between GP models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, (model_name, res) in zip(axes, results.items()):
        y_pred = res['y_pred'].ravel()
        y_std = res['y_std'].ravel()
        x_test = X_test.ravel()
        
        # Sort for plotting
        sort_idx = np.argsort(x_test)
        x_test = x_test[sort_idx]
        y_pred = y_pred[sort_idx]
        y_std = y_std[sort_idx]
        y_true = y_test_true.ravel()[sort_idx]
        
        ax.fill_between(x_test, y_pred - 2*y_std, y_pred + 2*y_std, 
                       alpha=0.3, label='±2σ')
        ax.plot(x_test, y_pred, 'b-', label='Prediction', linewidth=2)
        ax.plot(x_test, y_true, 'k--', label='True', linewidth=1.5)
        ax.scatter(X_train.ravel(), y_train.ravel(), c='r', s=30, 
                  zorder=5, label='Training data')
        
        ax.set_title(f'{model_name}\nRMSE={res["metrics"]["rmse"]:.4f}, '
                    f'R²={res["metrics"]["r2"]:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Function: {func_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_2d_comparison(X_train, y_train, X_test, y_test_true, 
                       results, func_name, save_path=None):
    """Plot 2D comparison between GP models."""
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
        y_pred = res['y_pred'].reshape(n_per_dim, n_per_dim)
        y_std = res['y_std'].reshape(n_per_dim, n_per_dim)
        
        # Prediction
        ax = axes[0, col]
        c = ax.contourf(X1, X2, y_pred, levels=20, cmap='viridis')
        ax.scatter(X_train[:, 0], X_train[:, 1], c='red', s=20, marker='x')
        ax.set_title(f'{model_name} Prediction\n'
                    f'RMSE={res["metrics"]["rmse"]:.4f}')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        plt.colorbar(c, ax=ax)
        
        # Uncertainty
        ax = axes[1, col]
        c = ax.contourf(X1, X2, y_std, levels=20, cmap='Reds')
        ax.scatter(X_train[:, 0], X_train[:, 1], c='blue', s=20, marker='x')
        ax.set_title(f'{model_name} Uncertainty (σ)')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        plt.colorbar(c, ax=ax)
    
    # Error plots for both models
    ax = axes[1, 0]
    ax.axis('off')  # Empty plot in bottom-left corner
    
    # Add error row
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    for col, (model_name, res) in enumerate(results.items()):
        ax = axes2[col]
        error = np.abs(res['y_pred'] - y_test_true).reshape(n_per_dim, n_per_dim)
        c = ax.contourf(X1, X2, error, levels=20, cmap='Reds')
        ax.scatter(X_train[:, 0], X_train[:, 1], c='blue', s=20, marker='x')
        ax.set_title(f'{model_name} |Error|')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        plt.colorbar(c, ax=ax)
    
    fig2.suptitle(f'Function: {func_name} - Absolute Errors', fontsize=14)
    plt.tight_layout()
    
    fig.suptitle(f'Function: {func_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        # Save error plot separately
        error_save_path = save_path.replace('.png', '_errors.png')
        fig2.savefig(error_save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_results_table(all_results):
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    header = f"{'Function':<20} {'Model':<15} {'RMSE':<10} {'R²':<10} {'MAE':<10} {'NLPD':<10} {'2σ Calib':<10} {'Train(s)':<10}"
    print(header)
    print("-" * 100)
    
    for (dim, func_name), results in all_results.items():
        func_label = f"{dim}D-{func_name}"
        for model_name, res in results.items():
            m = res['metrics']
            row = f"{func_label:<20} {model_name:<15} {m['rmse']:<10.4f} {m['r2']:<10.4f} {m['mae']:<10.4f} {m['nlpd']:<10.4f} {m['within_2sigma']:<10.2%} {m['train_time']:<10.3f}"
            print(row)
            func_label = ""  # Only print once per function
        print("-" * 100)


# =============================================================================
# Main Test Runner
# =============================================================================

def run_tests(functions=None, dims=None, n_train=50, n_test=200, 
              noise_std=0.01, kern_type='rbf', opt_hyp=True,
              n_frequencies=15, period=3.0, visualize=True, save_dir=None):
    """Run GP fitting tests.
    
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
            X_train, y_train, X_test, y_test, y_test_true = generate_data(
                func, n_train, n_test, dim, noise_std=noise_std
            )
            
            results = {}
            
            # Test NumpyGPModel
            print(f"\n--- NumpyGPModel ---")
            try:
                metrics, y_pred, y_std, gp = fit_and_evaluate(
                    NumpyGPModel, X_train, y_train, X_test, y_test_true,
                    kern_type=kern_type, opt_hyp=opt_hyp
                )
                results['NumpyGP'] = {
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_std': y_std,
                }
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  Train time: {metrics['train_time']:.3f}s")
                if hasattr(gp, 'hyp'):
                    print(f"  Hyperparameters: {gp.hyp[0]}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
            
            # Test ScalableGPModel
            print(f"\n--- ScalableGPModel ---")
            try:
                metrics, y_pred, y_std, gp = fit_and_evaluate(
                    ScalableGPModel, X_train, y_train, X_test, y_test_true,
                    kern_type=kern_type, opt_hyp=opt_hyp,
                    n_frequencies=n_frequencies, periods=period
                )
                results['ScalableGP'] = {
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_std': y_std,
                }
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  Train time: {metrics['train_time']:.3f}s")
                if hasattr(gp, 'hyp'):
                    print(f"  Hyperparameters: {gp.hyp[0]}")
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
                    save_path = os.path.join(save_dir, f'{dim}D_{func_name}.png')
                
                if dim == 1:
                    plot_1d_comparison(X_train, y_train, X_test, y_test_true,
                                      results, f'{dim}D {func_name}', save_path)
                elif dim == 2:
                    plot_2d_comparison(X_train, y_train, X_test, y_test_true,
                                      results, f'{dim}D {func_name}', save_path)
    
    # Print summary table
    print_results_table(all_results)
    
    return all_results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test GP fitting quality')
    parser.add_argument('--functions', type=str, default=None,
                       help='Comma-separated list of functions to test')
    parser.add_argument('--dims', type=str, default='1,2',
                       help='Comma-separated list of dimensions to test')
    parser.add_argument('--n-train', type=int, default=200,
                       help='Number of training points')
    parser.add_argument('--n-test', type=int, default=50,
                       help='Number of test points')
    parser.add_argument('--noise', type=float, default=0.01,
                       help='Observation noise std')
    parser.add_argument('--kernel', type=str, default='sum_lin_rbf',
                       help='Kernel type (rbf, mat52, sum_lin_rbf)')
    parser.add_argument('--no-opt', action='store_true',
                       help='Disable hyperparameter optimization')
    parser.add_argument('--n-freq', type=int, default=5,
                       help='Number of frequencies for ScalableGP')
    parser.add_argument('--period', type=float, default=10.0,
                       help='Period for ScalableGP')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    functions = args.functions.split(',') if args.functions else None
    dims = [int(d) for d in args.dims.split(',')]
    
    run_tests(
        functions=functions,
        dims=dims,
        n_train=args.n_train,
        n_test=args.n_test,
        noise_std=args.noise,
        kern_type=args.kernel,
        opt_hyp=not args.no_opt,
        n_frequencies=args.n_freq,
        period=args.period,
        visualize=not args.no_viz,
        save_dir=args.save_dir,
    )
