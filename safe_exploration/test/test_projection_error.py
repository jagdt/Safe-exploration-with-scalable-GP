#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for projection error computation."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from safe_exploration.ssm_numpy.scalable_gp import ScalableGPModel
from safe_exploration.ssm_numpy.gp_bounds import ScalableGPBounds


def test_projection_error_computation():
    """Test the ellipsoidal projection error computation."""
    
    print("=" * 80)
    print("Testing Ellipsoidal Projection Error Computation")
    print("=" * 80)
    
    # Create synthetic training data
    np.random.seed(42)
    n_train = 50
    n_s_in = 2
    n_u = 1
    n_s_out = 2
    
    X_train = np.random.randn(n_train, n_s_in + n_u) * 2.0
    y_train = np.sin(X_train[:, 0:1]) + 0.5 * X_train[:, 1:2] + 0.1 * np.random.randn(n_train, n_s_out)
    
    # Create and train scalable GP model
    print("\n1. Creating and training scalable GP model...")
    gp_model = ScalableGPModel(
        n_s_out=n_s_out,
        n_s_in=n_s_in,
        n_u=n_u,
        n_frequencies=5,
        periods=10.0,
        kern_types=["rbf"] * n_s_out,
        seed=42
    )
    
    gp_model.train(X_train, y_train)
    print("   Model trained successfully!")
    
    # Create bounds object
    print("\n2. Creating GP bounds object...")
    bounds = ScalableGPBounds(
        gp_model=gp_model,
        delta=0.05,
        rkhs_norm=None,  # Will be computed automatically
        R_subgaussian=1.0,
        projection_error=None  # Will use new ellipsoidal method
    )
    print("   Bounds object created successfully!")
    
    # Test projection error computation for each dimension
    print("\n3. Computing projection errors for each output dimension...")
    print("-" * 80)
    
    for dim_idx in range(n_s_out):
        print(f"\n--- Output Dimension {dim_idx} ---")
        
        # Compute ellipsoidal projection error
        print("\nEllipsoidal Projection Error:")
        ellips_error = bounds.compute_ellipsoidal_projection_error(dim_idx)
        
        # Compute old theoretical projection error for comparison
        print("\nTheoretical Projection Error (old method):")
        theory_error = bounds.compute_theoretical_projection_error(dim_idx)
        
        print(f"\nComparison:")
        print(f"  Ellipsoidal method: {ellips_error:.6f}")
        print(f"  Theoretical method: {theory_error:.6f}")
        print(f"  Ratio (Ellips/Theory): {ellips_error/theory_error:.4f}")
        
        print("-" * 80)
    
    # Test confidence bounds
    print("\n4. Testing confidence bounds computation...")
    X_test = np.random.randn(10, n_s_in + n_u) * 2.0
    
    for dim_idx in range(n_s_out):
        mu, sigma, lower, upper = bounds.confidence_bounds(X_test, dim_idx)
        print(f"\nDimension {dim_idx}:")
        print(f"  Mean range: [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  Std range: [{sigma.min():.4f}, {sigma.max():.4f}]")
        print(f"  Bound width (avg): {np.mean(upper - lower):.4f}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_projection_error_computation()
