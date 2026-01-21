#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for MPC exploration parameter sweeps
Aggregates results across multiple seeds and creates comparison plots
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'figure.figsize': (10, 7),
})

# RWTH colors
RWTH_BLUE = '#00549F'
RWTH_GREEN = '#57AB27'
RWTH_RED = '#A11035'
RWTH_ORANGE = '#F6A800'
RWTH_PURPLE = '#612158'
RWTH_TURQUOISE = '#0098A1'


def load_results_from_directory(results_dir):
    """
    Load all JSON result files from a directory
    
    Returns
    -------
    list of dicts: List of all loaded results
    """
    results_list = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Warning: Directory {results_dir} does not exist")
        return results_list
    
    # Find all .json files (except summary.json)
    json_files = [f for f in results_path.glob("*.json") if f.name != "summary.json"]
    print(f"Found {len(json_files)} result files in {results_dir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Convert lists back to numpy arrays where appropriate
                if 'information_gain' in data and isinstance(data['information_gain'], list):
                    data['information_gain'] = np.array(data['information_gain'])
                results_list.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results_list


def aggregate_by_parameters(results_list, param_names):
    """
    Aggregate results by specified parameters
    
    Parameters
    ----------
    results_list : list
        List of result dictionaries
    param_names : list
        List of parameter names to group by
        
    Returns
    -------
    dict: {(param_values): list of results with those param values}
    """
    grouped = defaultdict(list)
    
    for result in results_list:
        if not result.get('success', False):
            continue  # Skip failed experiments
        
        # Extract parameter values
        param_values = tuple(result['config'].get(pname, None) for pname in param_names)
        grouped[param_values].append(result)
    
    return dict(grouped)


def plot_timing_breakdown(results_by_type, gp_types, param_name, param_values, output_dir=None):
    """
    Create stacked bar chart showing timing breakdown by component
    
    Parameters
    ----------
    results_by_type : dict
        {gp_type: {param_value: [results]}}
    gp_types : list
        List of GP types to compare
    param_name : str
        Name of the parameter being varied
    param_values : list
        Sorted list of parameter values
    output_dir : str, optional
        Directory to save plots
    """
    fig, axes = plt.subplots(1, len(gp_types), figsize=(8 * len(gp_types), 6))
    if len(gp_types) == 1:
        axes = [axes]
    
    # Timing components to visualize
    components = ['mpc_optimization', 'gp_training', 'total']
    component_labels = ['MPC Optimization', 'GP Training', 'Other']
    component_colors = [RWTH_BLUE, RWTH_ORANGE, RWTH_GREEN]
    
    for ax_idx, gp_type in enumerate(gp_types):
        ax = axes[ax_idx]
        
        # Collect timing data
        timing_means = {comp: [] for comp in components}
        
        for param_val in param_values:
            if param_val not in results_by_type[gp_type]:
                for comp in components:
                    timing_means[comp].append(0.0)
                continue
            
            results_list = results_by_type[gp_type][param_val]
            
            # Average across seeds
            comp_times = {comp: [] for comp in components}
            for result in results_list:
                if 'timing' in result and result['timing']:
                    for comp in components:
                        if comp in result['timing']:
                            comp_times[comp].append(result['timing'][comp]['mean'])
            
            # Compute mean across seeds
            for comp in components:
                if comp_times[comp]:
                    timing_means[comp].append(np.mean(comp_times[comp]))
                else:
                    timing_means[comp].append(0.0)
        
        # Calculate 'other' time as total - mpc - gp_training
        other_time = []
        for i in range(len(param_values)):
            total = timing_means['total'][i]
            mpc = timing_means['mpc_optimization'][i]
            gp = timing_means['gp_training'][i]
            other = max(0.0, total - mpc - gp)  # Ensure non-negative
            other_time.append(other)
        
        # Build list of components to plot (exclude 'total', add 'other')
        plot_components = ['mpc_optimization', 'gp_training', 'other']
        plot_labels = ['MPC Optimization', 'GP Training', 'Other']
        plot_colors = [RWTH_BLUE, RWTH_ORANGE, RWTH_GREEN]
        plot_values = {
            'mpc_optimization': timing_means['mpc_optimization'],
            'gp_training': timing_means['gp_training'],
            'other': other_time
        }
        
        # Filter out components with all zeros (for cleaner visualization)
        active_components = []
        active_labels = []
        active_colors = []
        for comp, label, color in zip(plot_components, plot_labels, plot_colors):
            if np.sum(plot_values[comp]) > 1e-6:
                active_components.append(comp)
                active_labels.append(label)
                active_colors.append(color)
        
        # Create stacked bar chart
        x = np.arange(len(param_values))
        width = 0.6
        bottom = np.zeros(len(param_values))
        
        for comp, label, color in zip(active_components, active_labels, active_colors):
            values = np.array(plot_values[comp])
            ax.bar(x, values, width, label=label, bottom=bottom, color=color, alpha=0.8)
            bottom += values
        
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('Average Time per Iteration (s)')
        gp_label = 'Standard GP' if gp_type == 'numpy' else 'Scalable GP'
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in param_values])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        save_path = Path(output_dir) / f"timing_breakdown_{param_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_info_gain_trajectories(results_by_type, gp_types, param_name, param_values, output_dir=None):
    """
    Plot information gain development over iterations for different parameter values
    
    Parameters
    ----------
    results_by_type : dict
        {gp_type: {param_value: [results]}}
    gp_types : list
        List of GP types to compare
    param_name : str
        Name of the parameter being varied
    param_values : list
        Sorted list of parameter values
    output_dir : str, optional
        Directory to save plots
    """
    fig, axes = plt.subplots(1, len(gp_types), figsize=(8 * len(gp_types), 6))
    if len(gp_types) == 1:
        axes = [axes]
    
    # Color palette for different parameter values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(param_values)))
    
    for ax_idx, gp_type in enumerate(gp_types):
        ax = axes[ax_idx]
        
        for param_idx, param_val in enumerate(param_values):
            if param_val not in results_by_type[gp_type]:
                continue
            
            results_list = results_by_type[gp_type][param_val]
            
            # Extract information gain trajectories
            # Prefer reference information gain (ground truth hyperparameters) if available
            trajectories = []
            for result in results_list:
                # Try reference information gain first
                info_gain_key = 'information_gain_reference' if 'information_gain_reference' in result else 'information_gain'
                
                if info_gain_key in result:
                    # New format: dict with 'trajectory' or 'per_iteration'
                    if isinstance(result[info_gain_key], dict):
                        if 'trajectory' in result[info_gain_key]:
                            traj = result[info_gain_key]['trajectory']
                        elif 'per_iteration' in result[info_gain_key]:
                            traj = result[info_gain_key]['per_iteration']
                        else:
                            continue
                        if isinstance(traj, list):
                            traj = np.array(traj)
                        # Sum across dimensions if multi-dimensional
                        if traj.ndim > 1:
                            traj = np.sum(traj, axis=1)
                        trajectories.append(traj)
                    # Legacy format: array
                    elif isinstance(result['information_gain'], (list, np.ndarray)):
                        traj = np.array(result['information_gain'])
                        if traj.ndim > 1:
                            traj = np.sum(traj, axis=1)
                        trajectories.append(traj)
            
            if not trajectories:
                continue
            
            # Ensure all trajectories have the same length
            min_len = min(len(t) for t in trajectories)
            trajectories = [t[:min_len] for t in trajectories]
            trajectories = np.array(trajectories)
            
            # Calculate mean and std across seeds
            mean_traj = np.mean(trajectories, axis=0)
            std_traj = np.std(trajectories, axis=0)
            iterations = np.arange(1, len(mean_traj) + 1)
            
            # Plot with shaded error region
            label = f'{param_name.replace("_", " ").replace("n ", "N=").replace("N=safe samples", "N=")}{param_val}'
            ax.plot(iterations, mean_traj, color=colors[param_idx], label=label, linewidth=2.5)
            ax.fill_between(iterations, mean_traj - std_traj, mean_traj + std_traj, 
                          color=colors[param_idx], alpha=0.2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Information Gain')
        gp_label = 'Standard GP' if gp_type == 'numpy' else 'Scalable GP'
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        save_path = Path(output_dir) / f"info_gain_trajectory_{param_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_initial_samples_comparison(results_dir, output_dir=None):
    """
    Plot timing comparison between standard GP and scalable GP
    for varying number of initial samples
    """
    print("\n" + "="*80)
    print("EVALUATING INITIAL SAMPLES SWEEP")
    print("="*80 + "\n")
    
    # Load results
    results_list = load_results_from_directory(results_dir)
    if not results_list:
        print("No results found!")
        return
    
    # Group by (gp_type, n_safe_samples)
    grouped = aggregate_by_parameters(results_list, ['gp_type', 'n_safe_samples'])
    
    # Separate by GP type
    numpy_results = defaultdict(list)
    scalable_results = defaultdict(list)
    
    for (gp_type, n_samples), results in grouped.items():
        if gp_type == 'numpy':
            numpy_results[n_samples].extend(results)
        elif gp_type == 'scalable':
            scalable_results[n_samples].extend(results)
    
    # Get sorted n_safe_samples values
    n_samples_numpy = sorted(numpy_results.keys())
    n_samples_scalable = sorted(scalable_results.keys())
    n_samples_values = sorted(set(n_samples_numpy + n_samples_scalable))
    
    # Initialize arrays for statistics
    numpy_time_mean, numpy_time_std = [], []
    scalable_time_mean, scalable_time_std = [], []
    numpy_info_gain_mean, numpy_info_gain_std = [], []
    scalable_info_gain_mean, scalable_info_gain_std = [], []
    numpy_feasible_mean, numpy_feasible_std = [], []
    scalable_feasible_mean, scalable_feasible_std = [], []
    
    print("Aggregating results across seeds...")
    for n_samples in n_samples_values:
        print(f"\nn_safe_samples = {n_samples}:")
        
        # Standard GP
        if n_samples in numpy_results:
            results = numpy_results[n_samples]
            print(f"  Standard GP: {len(results)} runs")
            
            # Extract timing (total per iteration if available)
            times = []
            for r in results:
                if 'timing' in r and r['timing']:
                    # Check if it's the new format (dict with 'total')
                    if isinstance(r['timing'], dict) and 'total' in r['timing']:
                        times.append(r['timing']['total']['mean'])
                    # Legacy format
                    elif 'avg_iteration_time' in r['timing']:
                        times.append(r['timing']['avg_iteration_time'])
            
            if times:
                numpy_time_mean.append(np.mean(times))
                numpy_time_std.append(np.std(times))
            else:
                numpy_time_mean.append(np.nan)
                numpy_time_std.append(np.nan)
            
            # Extract information gain (final value)
            # Prefer reference information gain (ground truth hyperparameters) if available
            info_gains = []
            for r in results:
                # Try reference information gain first
                info_gain_key = 'information_gain_reference' if 'information_gain_reference' in r else 'information_gain'
                
                if info_gain_key in r:
                    # New format: dict with 'final'
                    if isinstance(r[info_gain_key], dict) and 'final' in r[info_gain_key]:
                        info_gains.append(r[info_gain_key]['final'])
                    # Legacy format: array
                    elif isinstance(r[info_gain_key], (list, np.ndarray)) and len(r[info_gain_key]) > 0:
                        info_gains.append(r[info_gain_key][-1])
            
            if info_gains:
                numpy_info_gain_mean.append(np.mean(info_gains))
                numpy_info_gain_std.append(np.std(info_gains))
            else:
                numpy_info_gain_mean.append(np.nan)
                numpy_info_gain_std.append(np.nan)
            
            # Extract feasibility
            feasible = []
            for r in results:
                if 'n_feasible_iterations' in r:
                    n_iters = r['config'].get('n_iterations', 20)
                    feasible.append(r['n_feasible_iterations'] / n_iters * 100)
            
            if feasible:
                numpy_feasible_mean.append(np.mean(feasible))
                numpy_feasible_std.append(np.std(feasible))
            else:
                numpy_feasible_mean.append(np.nan)
                numpy_feasible_std.append(np.nan)
        else:
            numpy_time_mean.append(np.nan)
            numpy_time_std.append(np.nan)
            numpy_info_gain_mean.append(np.nan)
            numpy_info_gain_std.append(np.nan)
            numpy_feasible_mean.append(np.nan)
            numpy_feasible_std.append(np.nan)
        
        # Scalable GP
        if n_samples in scalable_results:
            results = scalable_results[n_samples]
            print(f"  Scalable GP: {len(results)} runs")
            
            # Extract timing
            times = []
            for r in results:
                if 'timing' in r and r['timing']:
                    # Check if it's the new format (dict with 'total')
                    if isinstance(r['timing'], dict) and 'total' in r['timing']:
                        times.append(r['timing']['total']['mean'])
                    # Legacy format
                    elif 'avg_iteration_time' in r['timing']:
                        times.append(r['timing']['avg_iteration_time'])
            
            if times:
                scalable_time_mean.append(np.mean(times))
                scalable_time_std.append(np.std(times))
            else:
                scalable_time_mean.append(np.nan)
                scalable_time_std.append(np.nan)
            
            # Extract information gain
            # Prefer reference information gain (ground truth hyperparameters) if available
            info_gains = []
            for r in results:
                # Try reference information gain first
                info_gain_key = 'information_gain_reference' if 'information_gain_reference' in r else 'information_gain'
                
                if info_gain_key in r:
                    # New format: dict with 'final'
                    if isinstance(r[info_gain_key], dict) and 'final' in r[info_gain_key]:
                        info_gains.append(r[info_gain_key]['final'])
                    # Legacy format: array
                    elif isinstance(r[info_gain_key], (list, np.ndarray)) and len(r[info_gain_key]) > 0:
                        info_gains.append(r[info_gain_key][-1])
            
            if info_gains:
                scalable_info_gain_mean.append(np.mean(info_gains))
                scalable_info_gain_std.append(np.std(info_gains))
            else:
                scalable_info_gain_mean.append(np.nan)
                scalable_info_gain_std.append(np.nan)
            
            # Extract feasibility
            feasible = []
            for r in results:
                if 'n_feasible_iterations' in r:
                    n_iters = r['config'].get('n_iterations', 20)
                    feasible.append(r['n_feasible_iterations'] / n_iters * 100)
            
            if feasible:
                scalable_feasible_mean.append(np.mean(feasible))
                scalable_feasible_std.append(np.std(feasible))
            else:
                scalable_feasible_mean.append(np.nan)
                scalable_feasible_std.append(np.nan)
        else:
            scalable_time_mean.append(np.nan)
            scalable_time_std.append(np.nan)
            scalable_info_gain_mean.append(np.nan)
            scalable_info_gain_std.append(np.nan)
            scalable_feasible_mean.append(np.nan)
            scalable_feasible_std.append(np.nan)
    
    # Convert to numpy arrays
    n_samples_values = np.array(n_samples_values)
    numpy_time_mean = np.array(numpy_time_mean)
    numpy_time_std = np.array(numpy_time_std)
    scalable_time_mean = np.array(scalable_time_mean)
    scalable_time_std = np.array(scalable_time_std)
    numpy_info_gain_mean = np.array(numpy_info_gain_mean)
    numpy_info_gain_std = np.array(numpy_info_gain_std)
    scalable_info_gain_mean = np.array(scalable_info_gain_mean)
    scalable_info_gain_std = np.array(scalable_info_gain_std)
    numpy_feasible_mean = np.array(numpy_feasible_mean)
    numpy_feasible_std = np.array(numpy_feasible_std)
    scalable_feasible_mean = np.array(scalable_feasible_mean)
    scalable_feasible_std = np.array(scalable_feasible_std)
    
    # Plot 1: Timing comparison
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    valid_numpy = ~np.isnan(numpy_time_mean)
    valid_scalable = ~np.isnan(scalable_time_mean)
    
    if np.any(valid_numpy):
        ax1.errorbar(n_samples_values[valid_numpy], numpy_time_mean[valid_numpy], 
                   yerr=numpy_time_std[valid_numpy],
                   label='Standard GP', marker='o', capsize=5, color=RWTH_GREEN)
    if np.any(valid_scalable):
        ax1.errorbar(n_samples_values[valid_scalable], scalable_time_mean[valid_scalable],
                   yerr=scalable_time_std[valid_scalable],
                   label='Scalable GP', marker='s', capsize=5, color=RWTH_BLUE)
    ax1.set_xlabel('Number of Initial Samples')
    ax1.set_ylabel('Avg. Time per Iteration (s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure 1
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath1 = os.path.join(output_dir, "initial_samples_timing.png")
        plt.savefig(filepath1, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {filepath1}")
    
    plt.show()
    
    # Plot 2: Information gain
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    if np.any(valid_numpy):
        ax2.errorbar(n_samples_values[valid_numpy], numpy_info_gain_mean[valid_numpy],
                   yerr=numpy_info_gain_std[valid_numpy],
                   label='Standard GP', marker='o', capsize=5, color=RWTH_GREEN)
    if np.any(valid_scalable):
        ax2.errorbar(n_samples_values[valid_scalable], scalable_info_gain_mean[valid_scalable],
                   yerr=scalable_info_gain_std[valid_scalable],
                   label='Scalable GP', marker='s', capsize=5, color=RWTH_BLUE)
    ax2.set_xlabel('Number of Initial Samples')
    ax2.set_ylabel('Final Information Gain')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure 2
    if output_dir:
        filepath2 = os.path.join(output_dir, "initial_samples_info_gain.png")
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {filepath2}")
    
    plt.show()
    
    # Create timing breakdown visualization
    print("\nCreating timing breakdown visualization...")
    results_by_type = {
        'numpy': numpy_results,
        'scalable': scalable_results
    }
    plot_timing_breakdown(results_by_type, ['numpy', 'scalable'], 'n_safe_samples', n_samples_values, output_dir)
    
    # Create information gain trajectory plot
    print("\nCreating information gain trajectory plots...")
    plot_info_gain_trajectories(results_by_type, ['numpy', 'scalable'], 'n_safe_samples', n_samples_values.tolist(), output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: INITIAL SAMPLES SWEEP")
    print("="*80)
    for i, n_val in enumerate(n_samples_values):
        if not np.isnan(numpy_time_mean[i]) or not np.isnan(scalable_time_mean[i]):
            print(f"\nn_safe_samples = {n_val}:")
            if not np.isnan(numpy_time_mean[i]):
                print(f"  Standard GP:")
                print(f"    Time: {numpy_time_mean[i]:.3f} ± {numpy_time_std[i]:.3f} s")
                print(f"    Info gain: {numpy_info_gain_mean[i]:.3f} ± {numpy_info_gain_std[i]:.3f}")
                print(f"    Feasibility: {numpy_feasible_mean[i]:.1f} ± {numpy_feasible_std[i]:.1f} %")
            if not np.isnan(scalable_time_mean[i]):
                print(f"  Scalable GP:")
                print(f"    Time: {scalable_time_mean[i]:.3f} ± {scalable_time_std[i]:.3f} s")
                print(f"    Info gain: {scalable_info_gain_mean[i]:.3f} ± {scalable_info_gain_std[i]:.3f}")
                print(f"    Feasibility: {scalable_feasible_mean[i]:.1f} ± {scalable_feasible_std[i]:.1f} %")
            if not np.isnan(numpy_time_mean[i]) and not np.isnan(scalable_time_mean[i]):
                speedup = numpy_time_mean[i] / scalable_time_mean[i]
                print(f"    Speedup: {speedup:.2f}x")


def plot_frequency_sweep(results_dir, output_dir=None):
    """
    Plot projection error and information gain vs number of frequencies
    """
    print("\n" + "="*80)
    print("EVALUATING FREQUENCY SWEEP")
    print("="*80 + "\n")
    
    # Load results
    results_list = load_results_from_directory(results_dir)
    if not results_list:
        print("No results found!")
        return
    
    # Separate standard GP baseline from scalable GP results
    numpy_baseline = [r for r in results_list if r.get('config', {}).get('gp_type') == 'numpy']
    scalable_results = [r for r in results_list if r.get('config', {}).get('gp_type') == 'scalable']
    
    # Group scalable results by n_frequencies
    grouped = aggregate_by_parameters(scalable_results, ['n_frequencies'])
    
    # Sort by parameter value
    freq_values = sorted(grouped.keys())
    freq_values = [f[0] for f in freq_values]  # Extract from tuple
    
    # Calculate standard GP baseline statistics
    numpy_info_gain_mean, numpy_info_gain_std = np.nan, np.nan
    if numpy_baseline:
        print(f"\nStandard GP baseline: {len(numpy_baseline)} runs")
        numpy_info_gains = []
        for r in numpy_baseline:
            # Prefer reference information gain (ground truth hyperparameters)
            info_gain_key = 'information_gain_reference' if 'information_gain_reference' in r else 'information_gain'
            
            if info_gain_key in r:
                if isinstance(r[info_gain_key], dict) and 'final' in r[info_gain_key]:
                    numpy_info_gains.append(r[info_gain_key]['final'])
                elif isinstance(r[info_gain_key], (list, np.ndarray)) and len(r[info_gain_key]) > 0:
                    numpy_info_gains.append(r[info_gain_key][-1])
        
        if numpy_info_gains:
            numpy_info_gain_mean = np.mean(numpy_info_gains)
            numpy_info_gain_std = np.std(numpy_info_gains)
            print(f"  Info gain: {numpy_info_gain_mean:.3f} ± {numpy_info_gain_std:.3f}")
    
    # Initialize arrays for scalable GP results
    projection_error_mean, projection_error_std = [], []
    info_gain_mean, info_gain_std = [], []
    time_mean, time_std = [], []
    feasible_mean, feasible_std = [], []
    
    print("Aggregating results across seeds...")
    for n_freq in freq_values:
        results = grouped[(n_freq,)]
        print(f"\nM = {n_freq}: {len(results)} runs")
        
        # Projection error
        proj_errors = [r.get('projection_error', np.nan) for r in results]
        proj_errors = [e for e in proj_errors if not np.isnan(e)]
        if proj_errors:
            projection_error_mean.append(np.mean(proj_errors))
            projection_error_std.append(np.std(proj_errors))
        else:
            projection_error_mean.append(np.nan)
            projection_error_std.append(np.nan)
        
        # Information gain
        # Prefer reference information gain (ground truth hyperparameters) if available
        info_gains = []
        for r in results:
            info_gain_key = 'information_gain_reference' if 'information_gain_reference' in r else 'information_gain'
            
            if info_gain_key in r:
                # New format: dict with 'final'
                if isinstance(r[info_gain_key], dict) and 'final' in r[info_gain_key]:
                    info_gains.append(r[info_gain_key]['final'])
                # Legacy format: array
                elif isinstance(r[info_gain_key], (list, np.ndarray)) and len(r[info_gain_key]) > 0:
                    info_gains.append(r[info_gain_key][-1])
        
        if info_gains:
            info_gain_mean.append(np.mean(info_gains))
            info_gain_std.append(np.std(info_gains))
        else:
            info_gain_mean.append(np.nan)
            info_gain_std.append(np.nan)
        
        # Timing
        times = []
        for r in results:
            if 'timing' in r and r['timing']:
                # Check if it's the new format (dict with 'total')
                if isinstance(r['timing'], dict) and 'total' in r['timing']:
                    times.append(r['timing']['total']['mean'])
                # Legacy format
                elif 'avg_iteration_time' in r['timing']:
                    times.append(r['timing']['avg_iteration_time'])
        
        if times:
            time_mean.append(np.mean(times))
            time_std.append(np.std(times))
        else:
            time_mean.append(np.nan)
            time_std.append(np.nan)
        
        # Feasibility
        feasible = []
        for r in results:
            if 'n_feasible_iterations' in r:
                n_iters = r['config'].get('n_iterations', 20)
                feasible.append(r['n_feasible_iterations'] / n_iters * 100)
        
        if feasible:
            feasible_mean.append(np.mean(feasible))
            feasible_std.append(np.std(feasible))
        else:
            feasible_mean.append(np.nan)
            feasible_std.append(np.nan)
    
    # Convert to numpy arrays
    freq_values = np.array(freq_values)
    projection_error_mean = np.array(projection_error_mean)
    projection_error_std = np.array(projection_error_std)
    info_gain_mean = np.array(info_gain_mean)
    info_gain_std = np.array(info_gain_std)
    time_mean = np.array(time_mean)
    time_std = np.array(time_std)
    feasible_mean = np.array(feasible_mean)
    feasible_std = np.array(feasible_std)
    
    # Plot 1: Projection error
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    valid = ~np.isnan(projection_error_mean)
    if np.any(valid):
        ax1.errorbar(freq_values[valid], projection_error_mean[valid],
                   yerr=projection_error_std[valid],
                   marker='o', capsize=5, color=RWTH_RED)
        ax1.set_xlabel('Number of Frequencies')
        ax1.set_ylabel('Projection Error')
        ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure 1
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath1 = os.path.join(output_dir, "frequency_projection_error.png")
        plt.savefig(filepath1, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {filepath1}")
    
    plt.show()
    
    # Plot 2: Information gain
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    valid = ~np.isnan(info_gain_mean)
    if np.any(valid):
        ax2.errorbar(freq_values[valid], info_gain_mean[valid],
                   yerr=info_gain_std[valid],
                   marker='s', capsize=5, color=RWTH_BLUE, label='Scalable GP')
    
    # Add standard GP baseline as horizontal line
    if not np.isnan(numpy_info_gain_mean):
        ax2.axhline(y=numpy_info_gain_mean, color=RWTH_GREEN, linestyle='--', 
                   linewidth=2.5, label='Standard GP')
        # Add shaded region for standard deviation
        if not np.isnan(numpy_info_gain_std) and numpy_info_gain_std > 0:
            ax2.fill_between([min(freq_values), max(freq_values)],
                           numpy_info_gain_mean - numpy_info_gain_std,
                           numpy_info_gain_mean + numpy_info_gain_std,
                           color=RWTH_GREEN, alpha=0.2)
    
    ax2.set_xlabel('Number of Frequencies')
    ax2.set_ylabel('Final Information Gain')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure 2
    if output_dir:
        filepath2 = os.path.join(output_dir, "frequency_info_gain.png")
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {filepath2}")
    
    plt.show()
    
    # Plot 3: Timing
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    valid = ~np.isnan(time_mean)
    if np.any(valid):
        ax3.errorbar(freq_values[valid], time_mean[valid],
                   yerr=time_std[valid],
                   marker='D', capsize=5, color=RWTH_ORANGE)
        ax3.set_xlabel('Number of Frequencies')
        ax3.set_ylabel('Avg. Time per Iteration (s)')
        ax3.set_title('Computational Time vs Frequencies')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure 3
    if output_dir:
        filepath3 = os.path.join(output_dir, "frequency_timing.png")
        plt.savefig(filepath3, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {filepath3}")
    
    plt.show()
    
    # Create timing breakdown visualization
    print("\nCreating timing breakdown visualization...")
    scalable_results_dict = {}
    for (n_freq,), results in grouped.items():
        scalable_results_dict[n_freq] = results
    results_by_type = {'scalable': scalable_results_dict}
    plot_timing_breakdown(results_by_type, ['scalable'], 'n_frequencies', freq_values.tolist(), output_dir)
    
    # Create information gain trajectory plot
    print("\nCreating information gain trajectory plots...")
    plot_info_gain_trajectories(results_by_type, ['scalable'], 'n_frequencies', freq_values.tolist(), output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: FREQUENCY SWEEP")
    print("="*80)
    
    if not np.isnan(numpy_info_gain_mean):
        print(f"\nStandard GP (baseline):")
        print(f"  Info gain: {numpy_info_gain_mean:.3f} ± {numpy_info_gain_std:.3f}")
    
    print(f"\nScalable GP:")
    for i, freq in enumerate(freq_values):
        if not np.isnan(projection_error_mean[i]):
            print(f"\nM = {freq}:")
            print(f"  Projection error: {projection_error_mean[i]:.6f} ± {projection_error_std[i]:.6f}")
            print(f"  Info gain: {info_gain_mean[i]:.3f} ± {info_gain_std[i]:.3f}")
            print(f"  Time: {time_mean[i]:.3f} ± {time_std[i]:.3f} s")
            print(f"  Feasibility: {feasible_mean[i]:.1f} ± {feasible_std[i]:.1f} %")


def main():
    """Main evaluation function"""
    
    # Specify result directories
    timestamp = "20260121_103628"
    
    initial_samples_dir = f"experiments/results_exploration/initial_samples_sweep_{timestamp}"
    frequencies_dir = f"experiments/results_exploration/frequencies_sweep_{timestamp}"
    
    # Output directory for evaluation plots
    output_dir = f"experiments/results_exploration/evaluation_plots_{timestamp}"
    
    # Check if directories exist
    if os.path.exists(initial_samples_dir):
        print("Evaluating initial samples sweep...")
        plot_initial_samples_comparison(initial_samples_dir, output_dir)
    else:
        print(f"Initial samples results not found: {initial_samples_dir}")
        print("Please update the timestamp or run the sweep first.")
    
    if os.path.exists(frequencies_dir):
        print("\nEvaluating frequency sweep...")
        plot_frequency_sweep(frequencies_dir, output_dir)
    else:
        print(f"\nFrequency results not found: {frequencies_dir}")
        print("Please update the timestamp or run the sweep first.")


if __name__ == "__main__":
        main()
