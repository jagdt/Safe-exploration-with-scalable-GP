#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for debugging predict_casadi_symbolic in ScalableGPModel

This script creates a simple ScalableGP, trains it on synthetic data,
and tests the symbolic CasADi prediction against numerical prediction.
"""

import numpy as np
import sys
from casadi import SX, Function
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

from safe_exploration.ssm_numpy.scalable_gp import ScalableGPModel


def generate_synthetic_data(n_samples=50, n_s_in=2, n_u=1, seed=42):
    """Generate synthetic training data for testing"""
    np.random.seed(seed)
    
    # Generate random inputs
    X = np.random.randn(n_samples, n_s_in + n_u)
    
    # Simple nonlinear function for outputs
    y = np.zeros((n_samples, n_s_in))
    y[:, 0] = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + 0.1 * np.random.randn(n_samples)
    y[:, 1] = np.cos(X[:, 1]) * X[:, 2] + 0.1 * np.random.randn(n_samples)
    
    return X, y


def test_basic_prediction(gp, X_test):
    """Test basic numerical prediction"""
    print("\n" + "="*70)
    print("TEST 1: Basic Numerical Prediction")
    print("="*70)
    
    try:
        mu, sigma = gp.predict(X_test)
        print(f"✓ Numerical prediction successful")
        print(f"  Input shape: {X_test.shape}")
        print(f"  Mean shape: {mu.shape}")
        print(f"  Sigma shape: {sigma.shape}")
        print(f"  Mean: {mu}")
        print(f"  Sigma: {sigma}")
        return mu, sigma
    except Exception as e:
        print(f"✗ Numerical prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_casadi_symbolic_prediction(gp, X_test, n_s_in, n_u):
    """Test symbolic CasADi prediction"""
    print("\n" + "="*70)
    print("TEST 2: Symbolic CasADi Prediction (without gradients)")
    print("="*70)
    
    try:
        # Create symbolic variable
        x_sym = SX.sym("x_new", (1, n_s_in + n_u))
        print(f"✓ Created symbolic variable: {x_sym.shape}")
        
        # Get symbolic prediction
        mu_sym, sigma_sym = gp.predict_casadi_symbolic(x_sym, compute_grads=False)
        print(f"✓ Symbolic prediction successful (no grads)")
        print(f"  Symbolic mean shape: {mu_sym.shape}")
        print(f"  Symbolic sigma shape: {sigma_sym.shape}")
        
        # Create CasADi function
        f_pred = Function("f_pred", [x_sym], [mu_sym, sigma_sym], ["x"], ["mu", "sigma"])
        print(f"✓ Created CasADi function")
        
        # Evaluate at test point
        result = f_pred(x=X_test.T)
        mu_cas = np.array(result["mu"]).flatten()
        variance_cas = np.array(result["sigma"]).flatten()
        sigma_cas = np.sqrt(variance_cas)
        
        print(f"✓ CasADi evaluation successful")
        print(f"  Mean: {mu_cas}")
        print(f"  Sigma: {sigma_cas}")
        
        return mu_cas, sigma_cas, f_pred
    
    except Exception as e:
        print(f"✗ CasADi symbolic prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_casadi_with_gradients(gp, X_test, n_s_in, n_u):
    """Test symbolic CasADi prediction with gradients"""
    print("\n" + "="*70)
    print("TEST 3: Symbolic CasADi Prediction (with gradients)")
    print("="*70)
    
    try:
        # Create symbolic variable
        x_sym = SX.sym("x_new", (1, n_s_in + n_u))
        
        # Get symbolic prediction with gradients
        mu_sym, sigma_sym, jac_mu = gp.predict_casadi_symbolic(x_sym, compute_grads=True)
        print(f"✓ Symbolic prediction with gradients successful")
        print(f"  Symbolic mean shape: {mu_sym.shape}")
        print(f"  Symbolic sigma shape: {sigma_sym.shape}")
        print(f"  Symbolic jacobian shape: {jac_mu.shape}")
        
        # Create CasADi function
        f_pred_grad = Function("f_pred_grad", [x_sym], [mu_sym, sigma_sym, jac_mu], 
                              ["x"], ["mu", "sigma", "jac_mu"])
        print(f"✓ Created CasADi function with gradients")
        
        # Evaluate at test point
        result = f_pred_grad(x=X_test.T)
        mu_cas = np.array(result["mu"]).flatten()
        variance_cas = np.array(result["sigma"]).flatten()
        sigma_cas = np.sqrt(variance_cas)
        jac_cas = np.array(result["jac_mu"])
        
        print(f"✓ CasADi evaluation with gradients successful")
        print(f"  Mean: {mu_cas}")
        print(f"  Sigma: {sigma_cas}")
        print(f"  Jacobian shape: {jac_cas.shape}")
        print(f"  Jacobian:\n{jac_cas}")
        
        return mu_cas, sigma_cas, jac_cas
    
    except Exception as e:
        print(f"✗ CasADi prediction with gradients failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def compare_predictions(mu_numeric, sigma_numeric, mu_casadi, sigma_casadi):
    """Compare numerical and CasADi predictions"""
    print("\n" + "="*70)
    print("TEST 4: Comparison of Numerical vs CasADi Predictions")
    print("="*70)
    
    if mu_numeric is None or mu_casadi is None:
        print("✗ Cannot compare - one or both predictions failed")
        return
    
    # Ensure shapes match for comparison
    mu_numeric_flat = mu_numeric.flatten()
    sigma_numeric_flat = sigma_numeric.flatten()
    
    # Compare means
    mean_diff = np.abs(mu_numeric_flat - mu_casadi)
    mean_rel_diff = mean_diff / (np.abs(mu_numeric_flat) + 1e-10)
    
    print(f"Mean comparison:")
    print(f"  Numerical: {mu_numeric_flat}")
    print(f"  CasADi:    {mu_casadi}")
    print(f"  Absolute difference: {mean_diff}")
    print(f"  Relative difference: {mean_rel_diff}")
    print(f"  Max absolute difference: {np.max(mean_diff)}")
    print(f"  Max relative difference: {np.max(mean_rel_diff)}")
    
    # Compare sigmas
    sigma_diff = np.abs(sigma_numeric_flat - sigma_casadi)
    sigma_rel_diff = sigma_diff / (np.abs(sigma_numeric_flat) + 1e-10)
    
    print(f"\nSigma comparison:")
    print(f"  Numerical: {sigma_numeric_flat}")
    print(f"  CasADi:    {sigma_casadi}")
    print(f"  Absolute difference: {sigma_diff}")
    print(f"  Relative difference: {sigma_rel_diff}")
    print(f"  Max absolute difference: {np.max(sigma_diff)}")
    print(f"  Max relative difference: {np.max(sigma_rel_diff)}")
    
    # Check tolerances
    tol_abs = 1e-6
    tol_rel = 1e-4
    
    mean_match = np.all(mean_diff < tol_abs) or np.all(mean_rel_diff < tol_rel)
    sigma_match = np.all(sigma_diff < tol_abs) or np.all(sigma_rel_diff < tol_rel)
    
    print(f"\n{'✓' if mean_match else '✗'} Means match within tolerance")
    print(f"{'✓' if sigma_match else '✗'} Sigmas match within tolerance")
    
    return mean_match and sigma_match


def test_multiple_points(gp, n_s_in, n_u, n_points=5):
    """Test predictions on multiple random points"""
    print("\n" + "="*70)
    print(f"TEST 5: Multiple Point Predictions ({n_points} points)")
    print("="*70)
    
    all_match = True
    
    for i in range(n_points):
        print(f"\n--- Test point {i+1}/{n_points} ---")
        X_test = np.random.randn(1, n_s_in + n_u)
        
        # Numerical prediction
        mu_num, sigma_num = gp.predict(X_test)
        
        # CasADi prediction
        x_sym = SX.sym("x", (1, n_s_in + n_u))
        mu_sym, variance_sym = gp.predict_casadi_symbolic(x_sym, compute_grads=False)
        f = Function("f", [x_sym], [mu_sym, variance_sym])
        result = f(X_test.T)
        mu_cas = np.array(result[0]).flatten()
        variance_cas = np.array(result[1]).flatten()
        sigma_cas = np.sqrt(variance_cas)
        
        # Compare
        mean_diff = np.max(np.abs(mu_num.flatten() - mu_cas))
        sigma_diff = np.max(np.abs(sigma_num.flatten() - sigma_cas))
        
        match = mean_diff < 1e-5 and sigma_diff < 1e-5
        all_match = all_match and match
        
        status = "✓" if match else "✗"
        print(f"{status} Point {i+1}: mean_diff={mean_diff:.2e}, sigma_diff={sigma_diff:.2e}")
    
    print(f"\n{'✓' if all_match else '✗'} All {n_points} points match")
    return all_match


def visualize_predictions(gp, X_train, y_train, n_s_in, n_u):
    """Visualize predictions along a 1D slice (if applicable)"""
    print("\n" + "="*70)
    print("TEST 6: Visualization (1D slice through input space)")
    print("="*70)
    
    try:
        # Create a 1D grid varying the first input dimension
        n_grid = 100
        x_grid = np.linspace(-3, 3, n_grid)
        X_grid = np.zeros((n_grid, n_s_in + n_u))
        X_grid[:, 0] = x_grid
        
        # Numerical predictions
        mu_num, sigma_num = gp.predict(X_grid)
        
        # CasADi predictions
        x_sym = SX.sym("x", (1, n_s_in + n_u))
        mu_sym, variance_sym = gp.predict_casadi_symbolic(x_sym, compute_grads=False)
        f = Function("f", [x_sym], [mu_sym, variance_sym])
        
        mu_cas_list = []
        sigma_cas_list = []
        for i in range(n_grid):
            result = f(X_grid[i:i+1, :].T)
            mu_cas_list.append(np.array(result[0]).flatten())
            variance_cas = np.array(result[1]).flatten()
            sigma_cas_list.append(np.sqrt(variance_cas))  # Convert variance to std dev
        
        mu_cas = np.array(mu_cas_list)
        sigma_cas = np.array(sigma_cas_list)
        
        # Plot for first output dimension
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Mean comparison
        axes[0].plot(x_grid, mu_num[:, 0], 'b-', label='Numerical', linewidth=2)
        axes[0].plot(x_grid, mu_cas[:, 0], 'r--', label='CasADi', linewidth=2)
        axes[0].scatter(X_train[:, 0], y_train[:, 0], c='k', s=20, alpha=0.5, label='Training data')
        axes[0].set_ylabel('Mean prediction (dim 0)')
        axes[0].set_title('Predictive Mean Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Sigma comparison
        axes[1].plot(x_grid, sigma_num[:, 0], 'b-', label='Numerical', linewidth=2)
        axes[1].plot(x_grid, sigma_cas[:, 0], 'r--', label='CasADi', linewidth=2)
        axes[1].set_xlabel('Input dimension 0')
        axes[1].set_ylabel('Std deviation (dim 0)')
        axes[1].set_title('Predictive Uncertainty Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_predict_casadi_symbolic_visualization.png', dpi=150)
        print(f"✓ Visualization saved to 'test_predict_casadi_symbolic_visualization.png'")
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("="*70)
    print("Testing predict_casadi_symbolic in ScalableGPModel")
    print("="*70)
    
    # Configuration
    n_s_in = 2  # state dimensions
    n_u = 1     # control dimensions
    n_s_out = 2  # output dimensions
    n_samples = 50
    n_frequencies = 5
    
    print(f"\nConfiguration:")
    print(f"  State dimensions (n_s_in): {n_s_in}")
    print(f"  Control dimensions (n_u): {n_u}")
    print(f"  Output dimensions (n_s_out): {n_s_out}")
    print(f"  Training samples: {n_samples}")
    print(f"  Frequencies: {n_frequencies}")
    
    # Generate synthetic data
    print(f"\n{'='*70}")
    print("SETUP: Generating synthetic data and training GP")
    print('='*70)
    X_train, y_train = generate_synthetic_data(n_samples, n_s_in, n_u)
    print(f"✓ Generated {n_samples} training samples")
    
    # Create and train GP
    try:
        gp = ScalableGPModel(
            n_s_out=n_s_out,
            n_s_in=n_s_in,
            n_u=n_u,
            kern_types=['rbf'] * n_s_out,
            n_frequencies=n_frequencies,
            periods=10.0,
            train=False,
            seed=42
        )
        print(f"✓ Created ScalableGPModel")
        
        gp.train(X_train, y_train, opt_hyp=True)
        print(f"✓ Trained GP model")
        
    except Exception as e:
        print(f"✗ Failed to create/train GP: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create a test point
    X_test = np.array([[0.5, -0.3, 0.2]])
    print(f"\nTest point: {X_test}")
    
    # Run tests
    mu_numeric, sigma_numeric = test_basic_prediction(gp, X_test)
    
    mu_casadi, sigma_casadi, f_pred = test_casadi_symbolic_prediction(gp, X_test, n_s_in, n_u)
    
    mu_cas_grad, sigma_cas_grad, jac_cas = test_casadi_with_gradients(gp, X_test, n_s_in, n_u)
    
    if mu_numeric is not None and mu_casadi is not None:
        compare_predictions(mu_numeric, sigma_numeric, mu_casadi, sigma_casadi)
    
    test_multiple_points(gp, n_s_in, n_u, n_points=5)
    
    visualize_predictions(gp, X_train, y_train, n_s_in, n_u)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
