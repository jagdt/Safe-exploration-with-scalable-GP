#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parameter sweeps for MPC exploration experiments
Compare standard GP vs scalable GP with varying number of initial safe samples
"""

import sys
import os
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from safe_exploration.utils import make_json_serializable

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Change to experiments directory to ensure relative paths work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def run_exploration_experiment(config, experiment_name="experiment"):
    """
    Run a single exploration experiment with given configuration
    
    Parameters
    ----------
    config : Config
        Configuration object with all experiment parameters
    experiment_name : str
        Name for this experiment run
        
    Returns
    -------
    results : dict
        Dictionary with timing, information gain, and other metrics
    """
    from safe_exploration.exploration_runner import run_exploration
    
    print(f"\n{'='*80}")
    print(f"Running: {experiment_name}")
    print(f"  GP type: {config.gp_type}")
    print(f"  Initial samples: {config.n_safe_samples}")
    if config.gp_type == 'scalable':
        print(f"  Frequencies: {config.n_frequencies}")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Random seed: {config.seed}")
    print(f"{'='*80}\n")
    
    try:
        # Run exploration
        results = run_exploration(config)
        
        # Extract key metrics
        metrics = {
            'config': {
                'gp_type': config.gp_type,
                'n_safe_samples': config.n_safe_samples,
                'n_frequencies': getattr(config, 'n_frequencies', None),
                'n_iterations': config.n_iterations,
                'n_safe': config.n_safe,
                'seed': config.seed,
                'beta_safety': config.beta_safety,
            },
            'success': True,
            'experiment_name': experiment_name,
        }
        
        # Extract timing information from results
        if 'timing' in results and results['timing']:
            # results['timing'] is a list of dicts per experiment
            # Each dict contains arrays for different timing components
            timing_data = results['timing'][0] if len(results['timing']) > 0 else {}
            if timing_data:
                # Compute statistics for each timing component
                metrics['timing'] = {}
                for key, values in timing_data.items():
                    if isinstance(values, np.ndarray) and values.size > 0:
                        metrics['timing'][key] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'total': float(np.sum(values)),
                            'per_iteration': values.tolist()
                        }
        
        # Extract information gain
        if 'inf_gain' in results and results['inf_gain']:
            inf_gain = results['inf_gain'][0] if len(results['inf_gain']) > 0 else None
            if inf_gain is not None:
                # Sum across dimensions and get final value
                metrics['information_gain'] = {
                    'final': float(np.sum(inf_gain[-1])) if len(inf_gain) > 0 else 0.0,
                    'per_iteration': [float(np.sum(ig)) for ig in inf_gain]
                }
        
        # Extract reference information gain (with ground truth hyperparameters)
        if 'inf_gain_reference' in results and results['inf_gain_reference']:
            inf_gain_ref = results['inf_gain_reference'][0] if len(results['inf_gain_reference']) > 0 else None
            if inf_gain_ref is not None:
                metrics['information_gain_reference'] = {
                    'final': float(np.sum(inf_gain_ref[-1])) if len(inf_gain_ref) > 0 else 0.0,
                    'per_iteration': [float(np.sum(ig)) for ig in inf_gain_ref]
                }
        
        # Extract projection error (for scalable GP)
        if 'projection_error' in results and results['projection_error']:
            proj_err = results['projection_error'][0] if len(results['projection_error']) > 0 else None
            if proj_err is not None and not np.isnan(proj_err):
                metrics['projection_error'] = float(proj_err)
        
        # Extract GP hyperparameters
        if 'gp_hyperparameters' in results and results['gp_hyperparameters'] is not None:
            metrics['gp_hyperparameters'] = results['gp_hyperparameters']
        
        print(f"\n✓ Completed: {experiment_name}")
        
        return metrics
        
    except Exception as e:
        print(f"\n✗ Failed: {experiment_name}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'config': {
                'gp_type': config.gp_type,
                'n_safe_samples': config.n_safe_samples,
                'n_frequencies': getattr(config, 'n_frequencies', None),
                'n_iterations': config.n_iterations,
                'seed': config.seed,
            },
            'success': False,
            'error': str(e),
            'experiment_name': experiment_name,
        }


def sweep_initial_samples(n_values, seeds, base_overrides, output_dir):
    """
    Run experiments with varying number of initial safe samples
    Compare standard GP vs scalable GP
    
    Parameters
    ----------
    n_values : list
        List of initial sample counts to test
    seeds : list
        List of random seeds for reproducibility
    base_overrides : dict
        Configuration overrides to apply on top of default Config
    output_dir : str
        Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"SWEEPING INITIAL SAMPLES: {n_values}")
    print(f"COMPARING: Standard GP (numpy) vs Scalable GP")
    print(f"RANDOM SEEDS: {seeds}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for gp_type in ['numpy', 'scalable']:
        for n in n_values:
            for seed in seeds:
                # Create fresh config instance
                from journal_experiment_configs.dynamic_expl_pendulum_numpy import Config
                exp_config = Config()
                exp_config.gp_type = gp_type
                exp_config.n_safe_samples = n
                exp_config.seed = seed
                
                # Apply any additional base config overrides
                if base_overrides:
                    for key, value in base_overrides.items():
                        if key not in ['gp_type', 'n_safe_samples', 'seed']:
                            setattr(exp_config, key, value)
                
                exp_name = f"{gp_type}_n{n}_seed{seed}"
                exp_config.save_dir = f"{output_dir}/{exp_name}"
                exp_config.save_path_base = "."

                # Run experiment
                results = run_exploration_experiment(exp_config, exp_name)
                all_results.append(results)
                
                # Save intermediate results
                save_path = Path(output_dir) / f"{exp_name}.json"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    results_serializable = make_json_serializable(results)
                    json.dump(results_serializable, f, indent=2)
                
                print(f"Saved results to: {save_path}")
    
    # Save summary
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, 'w') as f:
        summary = {
            'sweep_type': 'initial_samples',
            'gp_types': ['numpy', 'scalable'],
            'n_values': n_values,
            'seeds': seeds,
            'n_experiments': len(all_results),
            'n_successful': sum(1 for r in all_results if r['success']),
        }
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"INITIAL SAMPLES SWEEP COMPLETE")
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful: {sum(1 for r in all_results if r['success'])}")
    print(f"Failed: {sum(1 for r in all_results if not r['success'])}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Main sweep execution"""
    
    # Base configuration overrides (only specify what differs from default Config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Common overrides for all experiments
    base_overrides = {
        'verbose': 1,
        'n_iterations': 50,
        'save_results': True,
        'save_vis': True,
        'visualize': False,
    }
    
    # Random seeds for statistical robustness
    seeds = [1,2]
    
    # Sweep: Vary initial samples, compare GP types
    print("\n" + "="*80)
    print("SWEEP: INITIAL SAMPLES (Standard GP vs Scalable GP)")
    print("="*80)
    
    n_values = [400, 600]
    overrides_samples = base_overrides.copy()
    overrides_samples['n_frequencies'] = 5  # Fixed for scalable GP in this sweep
    
    output_dir_samples = f"results_exploration/initial_samples_sweep_{timestamp}"
    sweep_initial_samples(n_values, seeds, overrides_samples, output_dir_samples)
    
    print("\n" + "="*80)
    print("SWEEP COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir_samples}")
    print(f"\nRun evaluate_exploration_sweeps.py to generate plots and analysis.")


if __name__ == "__main__":
    main()
