#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for MPC exploration initial samples sweep
Aggregates results across multiple seeds and creates comparison plots
"""

from logging import warning
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import json
from pathlib import Path
from collections import defaultdict
from matplotlib.ticker import MaxNLocator
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure matplotlib
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 4,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
    'lines.linewidth': 4.5,
    'text.usetex': False,
    'mathtext.fontset': 'cm'
})

# Larger font sizes for subfigure display (will be ~0.48 textwidth instead of 0.85)
large_plot_params = {
    'font.size': 30,
    'axes.labelsize': 36,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 28,
    'lines.linewidth': 4.5,
}

# RWTH colors
RWTH_BLUE = '#00549F'
RWTH_GREEN = '#57AB27'
RWTH_RED = '#A11035'
RWTH_ORANGE = '#F6A800'
RWTH_PURPLE = '#612158'
RWTH_TURQUOISE = '#0098A1'


def _rwth_light(hex_color, white_mix=0.2):
    """Return a lighter variant of a hex color by mixing with white."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    rgb_light = (1 - white_mix) * rgb + white_mix * np.ones(3)
    return mcolors.to_hex(rgb_light)


RWTH_LIGHT_BLUE = _rwth_light(RWTH_BLUE)
RWTH_LIGHT_GREEN = _rwth_light(RWTH_GREEN)


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
    Create grouped stacked bar chart showing timing breakdown by component
    
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
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Timing components to visualize
    components = ['initial_training', 'gp_training', 'mpc_optimization', 'total']
    
    # Collect timing data for all GP types
    all_timing_data = {}
    for gp_type in gp_types:
        timing_means = {comp: [] for comp in components}
        timing_stds = {comp: [] for comp in components}
        timing_raws = {comp: [] for comp in components}
        
        for param_val in param_values:
            if param_val not in results_by_type[gp_type]:
                for comp in components:
                    timing_means[comp].append(0.0)
                    timing_stds[comp].append(0.0)
                    timing_raws[comp].append([])
                continue
            
            results_list = results_by_type[gp_type][param_val]
            
            # Average across seeds
            comp_times = {comp: [] for comp in components}
            for result in results_list:
                if 'timing' in result and result['timing']:
                    n_iterations = result['config'].get('n_iterations', 20)
                    for comp in components:
                        if comp in result['timing']:
                            # Handle initial_training (scalar) vs per-iteration times (dict with 'mean')
                            if isinstance(result['timing'][comp], dict):
                                comp_times[comp].append(result['timing'][comp]['mean'])
                            else:
                                # For initial_training, amortize over all iterations (if it hasn't been amortized already)
                                comp_times[comp].append(result['timing'][comp] / n_iterations)
            
            # Compute mean and std across seeds
            for comp in components:
                if comp_times[comp]:
                    timing_means[comp].append(np.mean(comp_times[comp]))
                    timing_stds[comp].append(np.std(comp_times[comp]))
                    timing_raws[comp].append(comp_times[comp])
                else:
                    timing_means[comp].append(0.0)
                    timing_stds[comp].append(0.0)
                    timing_raws[comp].append([])
        
        # Calculate 'other' time as total - mpc - gp_training - initial_training
        other_time = []
        for i in range(len(param_values)):
            total = timing_means['total'][i]
            mpc = timing_means['mpc_optimization'][i]
            gp = timing_means['gp_training'][i]
            initial = timing_means['initial_training'][i]
            other = max(0.0, total - mpc - gp - initial)
            if other/total > 0.05:
                warning.warn(f"Large 'other' time detected for {gp_type} at param value {param_values[i]}")
            other_time.append(other)
        
        timing_means['other'] = other_time
        all_timing_data[gp_type] = {'means': timing_means, 'stds': timing_stds, 'raw_times': timing_raws}
    
    # Plot grouped stacked bars
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(param_values))
    width = 0.35
    
    # Store handles and labels for custom legend
    handles = []
    labels = []
    
    # Dummy handles for error bars and scatter points (will add at the end)
    std_dev_handle = None
    scatter_handle = None
    
    # Plot Standard GP (numpy)
    if 'numpy' in gp_types and 'numpy' in all_timing_data:
        bottom_numpy = np.zeros(len(param_values))
        
        # # Initial Training - solid, alpha=1.0
        # values = np.array(all_timing_data['numpy']['means']['initial_training'])
        # h1 = ax.bar(x - width/2, values, width,
        #        bottom=bottom_numpy, color=RWTH_GREEN, alpha=1.0)
        # handles.append(h1)
        # labels.append('Standard GP (Initial Training)')
        # bottom_numpy += values
        
        # GP Training - solid, alpha=0.8
        values = np.array(all_timing_data['numpy']['means']['gp_training'])
        h2 = ax.bar(x - width/2, values, width,
               bottom=bottom_numpy, color=RWTH_GREEN, alpha=0.8)
        handles.append(h2)
        labels.append('Standard GP (Online Training)')
        bottom_numpy += values
        
        # MPC Optimization - hatch, alpha=0.6 for color, alpha=1.0 for hatch
        values = np.array(all_timing_data['numpy']['means']['mpc_optimization'])
        # First layer: colored background with transparency
        h3a = ax.bar(x - width/2, values, width,
               bottom=bottom_numpy, color=RWTH_GREEN, alpha=0.6, linewidth=0)
        # Second layer: opaque black hatch lines only
        h3b = ax.bar(x - width/2, values, width,
               bottom=bottom_numpy, color='none', alpha=1.0, hatch='//', edgecolor='black', linewidth=0)
        handles.append((h3a, h3b))
        labels.append('Standard GP (MPC)')
        bottom_numpy += values
        
        # # Other time - very light color
        # values = np.array(all_timing_data['numpy']['means']['other'])
        # h4 = ax.bar(x - width/2, values, width,
        #        bottom=bottom_numpy, color=RWTH_GREEN, alpha=0.3)
        # handles.append(h4)
        # labels.append('Standard GP (Other)')
        # bottom_numpy += values
        
        # Add std dev bar at top of stack
        combined_means = np.array(all_timing_data['numpy']['means']['gp_training']) + np.array(all_timing_data['numpy']['means']['mpc_optimization'])
        combined_stds = np.array(all_timing_data['numpy']['stds']['gp_training']) + np.array(all_timing_data['numpy']['stds']['mpc_optimization'])
        eb = ax.errorbar(x - width/2, combined_means, yerr=combined_stds, fmt='none', ecolor='black', capsize=5, capthick=2, elinewidth=2, alpha=0.7, zorder=10)
        if std_dev_handle is None:
            std_dev_handle = eb
    
        # Plot individual total-time points to show spread as a vertical stack
        for idx in range(len(all_timing_data['numpy']['raw_times']['gp_training'])):
            gp_training = all_timing_data['numpy']['raw_times']['gp_training'][idx]
            mpc = all_timing_data['numpy']['raw_times']['mpc_optimization'][idx]
            times = [x + y for x, y in zip(gp_training, mpc)]
            if not times:
                continue
            x_positions = np.full(len(times), x[idx] - width/2)
            sc = ax.scatter(x_positions, times, color=_rwth_light(RWTH_GREEN,0.75), s=100, edgecolors=_rwth_light(RWTH_GREEN,0.25), linewidth=1.2, zorder=5)
            if scatter_handle is None:
                scatter_handle = sc
    
    # Plot Scalable GP
    if 'scalable' in gp_types and 'scalable' in all_timing_data:
        bottom_scalable = np.zeros(len(param_values))
        
        # # Initial Training - solid, alpha=1.0
        # values = np.array(all_timing_data['scalable']['means']['initial_training'])
        # h5 = ax.bar(x + width/2, values, width,
        #        bottom=bottom_scalable, color=RWTH_BLUE, alpha=1.0)
        # handles.append(h5)
        # labels.append('Scalable GP (Initial Training)')
        # bottom_scalable += values
        
        # GP Training - solid, alpha=0.8
        values = np.array(all_timing_data['scalable']['means']['gp_training'])
        h6 = ax.bar(x + width/2, values, width,
               bottom=bottom_scalable, color=RWTH_BLUE, alpha=0.8)
        handles.append(h6)
        labels.append('Scalable GP (Online Training)')
        bottom_scalable += values
        
        # MPC Optimization - hatch, alpha=0.6 for color, alpha=1.0 for hatch
        values = np.array(all_timing_data['scalable']['means']['mpc_optimization'])
        # First layer: colored background with transparency
        h7a = ax.bar(x + width/2, values, width,
               bottom=bottom_scalable, color=RWTH_BLUE, alpha=0.6, linewidth=0)
        # Second layer: opaque black hatch lines only
        h7b = ax.bar(x + width/2, values, width,
               bottom=bottom_scalable, color='none', alpha=1.0, hatch='//', edgecolor='black', linewidth=0)
        handles.append((h7a, h7b))
        labels.append('Scalable GP (MPC)')
        bottom_scalable += values
        
        # # Other time - very light color
        # values = np.array(all_timing_data['scalable']['means']['other'])
        # h8 = ax.bar(x + width/2, values, width,
        #        bottom=bottom_scalable, color=RWTH_BLUE, alpha=0.3)
        # handles.append(h8)
        # labels.append('Scalable GP (Other)')
        # bottom_scalable += values
        
        # Add std dev bar at top of stack
        combined_means = np.array(all_timing_data['scalable']['means']['gp_training']) + np.array(all_timing_data['scalable']['means']['mpc_optimization'])
        combined_stds = np.array(all_timing_data['scalable']['stds']['gp_training']) + np.array(all_timing_data['scalable']['stds']['mpc_optimization'])
        ax.errorbar(x + width/2, combined_means, yerr=combined_stds, fmt='none', ecolor='black', capsize=5, capthick=2, elinewidth=2, alpha=0.7, zorder=10)
    
        for idx in range(len(all_timing_data['scalable']['raw_times']['gp_training'])):
            gp_training = all_timing_data['scalable']['raw_times']['gp_training'][idx]
            mpc = all_timing_data['scalable']['raw_times']['mpc_optimization'][idx]
            times = [x + y for x, y in zip(gp_training, mpc)]
            if not times:
                continue
            x_positions = np.full(len(times), x[idx] + width/2)
            ax.scatter(x_positions, times, color=_rwth_light(RWTH_BLUE,0.75), s=100, edgecolors=_rwth_light(RWTH_BLUE,0.25), linewidth=1.2, zorder=5)
    
    ax.set_xlabel(r'Number of initial training points $N_{\mathrm{init}}$')
    ax.set_ylabel('Average time per iteration (s)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in param_values])
    
    # Add legend entries for std dev and scatter points
    if std_dev_handle is not None:
        handles.append(std_dev_handle)
        labels.append('Std. Dev. (total time)')
    
    # Create dummy scatter for legend with neutral color (represents both green and blue points in plot)
    dummy_scatter = ax.scatter([], [], color='gray', s=100, edgecolors='dimgray', linewidth=1.2, alpha=0.6)
    handles.append(dummy_scatter)
    labels.append('Individual Run Times')
    
    ax.legend(handles, labels, loc='upper right')
    
    # Add margin at top (10% extra space) and ensure y-axis starts at 0
    y_min, y_max = ax.get_ylim()
    ax.set_ylim([max(0, y_min), y_max * 1.1])
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"timing_breakdown_{param_name}.svg")
        plt.savefig(filepath, bbox_inches='tight')
        print(f"\nFigure saved to: {filepath}")
    
    plt.show()


def plot_info_gain_trajectories(results_by_type, gp_types, param_name, param_values, output_dir=None, use_large_fonts=True):
    """
    Plot mutual information development over iterations for different parameter values
    
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
    use_large_fonts : bool, optional
        Whether to use larger font sizes for subfigure display (default: True)
    """
    if use_large_fonts:
        # Larger font sizes for subfigure display (will be ~0.48 textwidth instead of 0.85)
        with plt.rc_context(large_plot_params):
            plot_info_gain_trajectories_impl(results_by_type, gp_types, param_name, param_values, output_dir)
    else:
        plot_info_gain_trajectories_impl(results_by_type, gp_types, param_name, param_values, output_dir)


def plot_info_gain_trajectories_impl(results_by_type, gp_types, param_name, param_values, output_dir=None):
    """Implementation of plot_info_gain_trajectories with custom font sizes applied."""
    # RWTH color palette for different parameter values
    rwth_colors = [
        RWTH_BLUE,      # #00549F
        RWTH_GREEN,     # #57AB27
        RWTH_ORANGE,    # #F6A800
        RWTH_RED,       # #A11035
        RWTH_PURPLE,    # #612158
        RWTH_TURQUOISE, # #0098A1
    ]
    
    # Extend if needed
    if len(param_values) > len(rwth_colors):
        # Use colormap for additional colors
        print("Extending color palette for additional parameter values")
        extra_colors = plt.cm.Set2(np.linspace(0, 1, len(param_values) - len(rwth_colors)))
        colors = rwth_colors + [tuple(c) for c in extra_colors]
    else:
        colors = rwth_colors[:len(param_values)]
    
    # Store figures and axes for shared y-axis limits
    figures_axes = []
    
    for gp_type in gp_types:
        # Create separate figure for each GP type
        fig, ax = plt.subplots(figsize=(12, 8))
        
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
            label = rf'$N_{{\mathrm{{init}}}}={param_val}$'
            ax.plot(iterations, mean_traj, color=colors[param_idx], label=label)
            ax.fill_between(iterations, mean_traj - std_traj, mean_traj + std_traj, 
                          color=colors[param_idx], alpha=0.2)
        
        ax.set_xlabel('Iteration', labelpad=10)
        ax.set_ylabel('Mutual information', labelpad=15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='upper left')
        # ax.grid(True, alpha=0.3)
        
        figures_axes.append((fig, ax, gp_type))
    
    # Set same y-axis limits for all plots
    if figures_axes:
        all_ylims = [ax.get_ylim() for fig, ax, gp_type in figures_axes]
        y_min = min(ylim[0] for ylim in all_ylims)
        y_max = max(ylim[1] for ylim in all_ylims)
        
        for fig, ax, gp_type in figures_axes:
            ax.set_ylim([y_min, y_max])
            plt.figure(fig.number)
            plt.tight_layout()
            
            if output_dir:
                filepath = os.path.join(output_dir, f"info_gain_trajectory_{gp_type}_{param_name}.svg")
                plt.savefig(filepath, bbox_inches='tight')
                print(f"Figure saved to: {filepath}")
            
            plt.show()


def plot_info_gain_comparison(results_by_type, gp_types, param_name, param_values, output_dir=None):
    """
    Plot mutual information comparison between Standard GP and Scalable GP in one plot
    Standard GP in blue (different shades), Scalable GP in green (different shades)
    
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
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Generate color shades: Standard GP in blue, Scalable GP in green
    n_params = len(param_values)

    def _shades_from_base(hex_color, n, white_mix_max=0.60):
        """Return n shades from light (mixed with white) to base color."""
        base_rgb = np.array(mcolors.to_rgb(hex_color))
        if n <= 1:
            return [base_rgb]
        shades = []
        for i in range(n):
            t = i / (n - 1)
            white_mix = white_mix_max * (1 - t)
            rgb = (1 - white_mix) * base_rgb + white_mix * np.ones(3)
            shades.append(rgb)
        return shades

    numpy_colors = _shades_from_base(RWTH_GREEN, n_params)
    scalable_colors = _shades_from_base(RWTH_BLUE, n_params)
    
    gp_colors = {
        'numpy': numpy_colors,
        'scalable': scalable_colors
    }
    
    # Define line styles for different parameter values
    line_styles = ['-', '--', '-.', ':']
    if n_params > len(line_styles):
        # Repeat line styles if needed
        line_styles = line_styles * ((n_params // len(line_styles)) + 1)
    
    gp_labels = {
        'numpy': 'Standard GP',
        'scalable': 'Scalable GP'
    }
    
    for param_idx, param_val in enumerate(param_values):
        for gp_type in gp_types:
            if param_val not in results_by_type[gp_type]:
                continue
            
            results_list = results_by_type[gp_type][param_val]
            
            # Extract information gain trajectories
            trajectories = []
            for result in results_list:
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
                    elif isinstance(result[info_gain_key], (list, np.ndarray)):
                        traj = np.array(result[info_gain_key])
                        if traj.ndim > 1:
                            traj = np.sum(traj, axis=1)
                        trajectories.append(traj)
            
            if not trajectories:
                continue
            
            # Ensure all trajectories have the same length
            min_len = min(len(t) for t in trajectories)
            trajectories = [t[:min_len] for t in trajectories]
            trajectories = np.array(trajectories)
            
            # Calculate mean across seeds (no std plotted)
            mean_traj = np.mean(trajectories, axis=0)
            iterations = np.arange(1, len(mean_traj) + 1)

            # Plot mean trajectory with specific line style
            label = rf'{gp_labels[gp_type]} ($N_{{\mathrm{{init}}}}={param_val}$)'
            ax.plot(
                iterations,
                mean_traj,
                color=gp_colors[gp_type][param_idx],
                linestyle=line_styles[param_idx],
                label=label,
                linewidth=3.5,
            )
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mutual information')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper left')
    # ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, f"info_gain_comparison_{param_name}.svg")
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
    
    plt.show()


def plot_safety_metrics(n_samples_values, 
                       numpy_safety_mean, numpy_safety_std,
                       scalable_safety_mean, scalable_safety_std,
                       numpy_inside_mean, numpy_inside_std,
                       scalable_inside_mean, scalable_inside_std,
                       output_dir=None,
                       use_large_fonts=True):
    """
    Plot safety metrics comparison between standard GP and scalable GP
    
    Parameters
    ----------
    n_samples_values : array
        Number of initial samples for each experiment
    numpy_safety_mean, numpy_safety_std : arrays
        Mean and std of safety verification success rate for numpy GP
    scalable_safety_mean, scalable_safety_std : arrays
        Mean and std of safety verification success rate for scalable GP
    numpy_inside_mean, numpy_inside_std : arrays
        Mean and std of trajectory inside ellipsoid rate for numpy GP
    scalable_inside_mean, scalable_inside_std : arrays
        Mean and std of trajectory inside ellipsoid rate for scalable GP
    output_dir : str, optional
        Directory to save plots
    use_large_fonts : bool, optional
        Whether to use larger font sizes for subfigure display (default: True)
    """
    if use_large_fonts:
        with plt.rc_context(large_plot_params):
            _plot_safety_metrics_impl(n_samples_values, numpy_safety_mean, numpy_safety_std,
                                      scalable_safety_mean, scalable_safety_std,
                                      numpy_inside_mean, numpy_inside_std,
                                      scalable_inside_mean, scalable_inside_std,
                                      output_dir)
    else:
        _plot_safety_metrics_impl(n_samples_values, numpy_safety_mean, numpy_safety_std,
                                  scalable_safety_mean, scalable_safety_std,
                                  numpy_inside_mean, numpy_inside_std,
                                  scalable_inside_mean, scalable_inside_std,
                                  output_dir)


def _plot_safety_metrics_impl(n_samples_values, 
                              numpy_safety_mean, numpy_safety_std,
                              scalable_safety_mean, scalable_safety_std,
                              numpy_inside_mean, numpy_inside_std,
                              scalable_inside_mean, scalable_inside_std,
                              output_dir=None):
    """Implementation of plot_safety_metrics."""
    x = np.arange(len(n_samples_values))
    width = 0.35
    
    # Plot 1: Safety verification success rate
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    ax1.bar(x - width/2, numpy_safety_mean, width, 
            label='Standard GP', 
            color=RWTH_LIGHT_GREEN, alpha=1.0, zorder=3)
    ax1.bar(x + width/2, scalable_safety_mean, width,
            label='Scalable GP',
            color=RWTH_LIGHT_BLUE, alpha=1.0, zorder=3)
    
    # Add grey line at 95% confidence level (on top of bars)
    ax1.axhline(y=95, color='grey', linestyle='--', linewidth=6, alpha=1.0, zorder=4, label='95% confidence level')
    
    ax1.set_xlabel(r'Number of initial training points $N_{\mathrm{init}}$', labelpad=10, x=0.42)
    ax1.set_ylabel('Safety verification\nsuccess rate (%)', labelpad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(n)) for n in n_samples_values])
    ax1.legend(loc='lower left')
    ax1.set_ylim([90, 101])
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "safety_verification_comparison.svg")
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
    
    plt.show()
    
    # Plot 2: Trajectory fully inside ellipsoid rate
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    ax2.bar(x - width/2, numpy_inside_mean, width,
            label='Standard GP',
            color=RWTH_LIGHT_GREEN, alpha=1.0, zorder=3)
    ax2.bar(x + width/2, scalable_inside_mean, width,
            label='Scalable GP',
            color=RWTH_LIGHT_BLUE, alpha=1.0, zorder=3)
    
    # Add grey line at 95% confidence level (on top of bars)
    ax2.axhline(y=95, color='grey', linestyle='--', linewidth=6, alpha=1.0, zorder=4, label='95% confidence level')
    
    ax2.set_xlabel(r'Number of initial training points $N_{\mathrm{init}}$', labelpad=10, x=0.42)
    ax2.set_ylabel('Trajectory inside\nellipsoid rate (%)', labelpad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(n)) for n in n_samples_values])
    ax2.legend(loc='lower left')
    ax2.set_ylim([90, 101])
    
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, "inside_ellipsoid_comparison.svg")
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
    
    plt.show()


def plot_initial_samples_comparison(results_dir, output_dir=None, use_large_fonts=True):
    """
    Plot timing comparison between standard GP and scalable GP
    for varying number of initial samples
    
    Parameters
    ----------
    results_dir : str
        Directory containing result files
    output_dir : str, optional
        Directory to save plots
    use_large_fonts : bool, optional
        Whether to use larger font sizes for subfigure display (default: True)
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
    numpy_safety_mean, numpy_safety_std = [], []
    scalable_safety_mean, scalable_safety_std = [], []
    numpy_inside_ellipsoid_mean, numpy_inside_ellipsoid_std = [], []
    scalable_inside_ellipsoid_mean, scalable_inside_ellipsoid_std = [], []
    
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
            
            # Extract safety metrics
            safety_rates = []
            inside_ellipsoid_rates = []
            for r in results:
                if 'safety_all' in r and r['safety_all'] is not None:
                    safety_all = np.array(r['safety_all'])
                    if len(safety_all) > 0:
                        safety_rates.append(np.mean(safety_all) * 100)  # Percentage
                
                if 'inside_ellipsoid' in r and r['inside_ellipsoid'] is not None:
                    inside_ellipsoid = np.array(r['inside_ellipsoid'])
                    if len(inside_ellipsoid) > 0:
                        # Check if all timesteps are inside ellipsoid for each iteration
                        all_inside = np.all(inside_ellipsoid, axis=1) if inside_ellipsoid.ndim > 1 else inside_ellipsoid
                        inside_ellipsoid_rates.append(np.mean(all_inside) * 100)  # Percentage
            
            if safety_rates:
                numpy_safety_mean.append(np.mean(safety_rates))
                numpy_safety_std.append(np.std(safety_rates))
            else:
                numpy_safety_mean.append(np.nan)
                numpy_safety_std.append(np.nan)
            
            if inside_ellipsoid_rates:
                numpy_inside_ellipsoid_mean.append(np.mean(inside_ellipsoid_rates))
                numpy_inside_ellipsoid_std.append(np.std(inside_ellipsoid_rates))
            else:
                numpy_inside_ellipsoid_mean.append(np.nan)
                numpy_inside_ellipsoid_std.append(np.nan)
        else:
            numpy_time_mean.append(np.nan)
            numpy_time_std.append(np.nan)
            numpy_info_gain_mean.append(np.nan)
            numpy_info_gain_std.append(np.nan)
            numpy_feasible_mean.append(np.nan)
            numpy_feasible_std.append(np.nan)
            numpy_safety_mean.append(np.nan)
            numpy_safety_std.append(np.nan)
            numpy_inside_ellipsoid_mean.append(np.nan)
            numpy_inside_ellipsoid_std.append(np.nan)
        
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
            
            # Extract safety metrics
            safety_rates = []
            inside_ellipsoid_rates = []
            for r in results:
                if 'safety_all' in r and r['safety_all'] is not None:
                    safety_all = np.array(r['safety_all'])
                    if len(safety_all) > 0:
                        safety_rates.append(np.mean(safety_all) * 100)  # Percentage
                
                if 'inside_ellipsoid' in r and r['inside_ellipsoid'] is not None:
                    inside_ellipsoid = np.array(r['inside_ellipsoid'])
                    if len(inside_ellipsoid) > 0:
                        # Check if all timesteps are inside ellipsoid for each iteration
                        all_inside = np.all(inside_ellipsoid, axis=1) if inside_ellipsoid.ndim > 1 else inside_ellipsoid
                        inside_ellipsoid_rates.append(np.mean(all_inside) * 100)  # Percentage
            
            if safety_rates:
                scalable_safety_mean.append(np.mean(safety_rates))
                scalable_safety_std.append(np.std(safety_rates))
            else:
                scalable_safety_mean.append(np.nan)
                scalable_safety_std.append(np.nan)
            
            if inside_ellipsoid_rates:
                scalable_inside_ellipsoid_mean.append(np.mean(inside_ellipsoid_rates))
                scalable_inside_ellipsoid_std.append(np.std(inside_ellipsoid_rates))
            else:
                scalable_inside_ellipsoid_mean.append(np.nan)
                scalable_inside_ellipsoid_std.append(np.nan)
        else:
            scalable_time_mean.append(np.nan)
            scalable_time_std.append(np.nan)
            scalable_info_gain_mean.append(np.nan)
            scalable_info_gain_std.append(np.nan)
            scalable_feasible_mean.append(np.nan)
            scalable_feasible_std.append(np.nan)
            scalable_safety_mean.append(np.nan)
            scalable_safety_std.append(np.nan)
            scalable_inside_ellipsoid_mean.append(np.nan)
            scalable_inside_ellipsoid_std.append(np.nan)
    
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
    numpy_safety_mean = np.array(numpy_safety_mean)
    numpy_safety_std = np.array(numpy_safety_std)
    scalable_safety_mean = np.array(scalable_safety_mean)
    scalable_safety_std = np.array(scalable_safety_std)
    numpy_inside_ellipsoid_mean = np.array(numpy_inside_ellipsoid_mean)
    numpy_inside_ellipsoid_std = np.array(numpy_inside_ellipsoid_std)
    scalable_inside_ellipsoid_mean = np.array(scalable_inside_ellipsoid_mean)
    scalable_inside_ellipsoid_std = np.array(scalable_inside_ellipsoid_std)
    
    # Create timing breakdown visualization
    print("\nCreating timing breakdown visualization...")
    results_by_type = {
        'numpy': numpy_results,
        'scalable': scalable_results
    }
    plot_timing_breakdown(results_by_type, ['numpy', 'scalable'], 'n_safe_samples', n_samples_values, output_dir)
    
    # Create mutual information trajectory plot
    print("\nCreating mutual information trajectory plots...")
    plot_info_gain_trajectories(results_by_type, ['numpy', 'scalable'], 'n_safe_samples', n_samples_values.tolist(), output_dir, use_large_fonts)
    
    # Create mutual information comparison plot
    print("\nCreating mutual information comparison plot...")
    plot_info_gain_comparison(results_by_type, ['numpy', 'scalable'], 'n_safe_samples', n_samples_values.tolist(), output_dir)
    
    # Create safety metrics plot
    print("\nCreating safety metrics comparison...")
    plot_safety_metrics(n_samples_values, 
                       numpy_safety_mean, numpy_safety_std,
                       scalable_safety_mean, scalable_safety_std,
                       numpy_inside_ellipsoid_mean, numpy_inside_ellipsoid_std,
                       scalable_inside_ellipsoid_mean, scalable_inside_ellipsoid_std,
                       output_dir, use_large_fonts)
    
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
                print(f"    Mutual information: {numpy_info_gain_mean[i]:.3f} ± {numpy_info_gain_std[i]:.3f}")
                print(f"    Feasibility: {numpy_feasible_mean[i]:.1f} ± {numpy_feasible_std[i]:.1f} %")
            if not np.isnan(scalable_time_mean[i]):
                print(f"  Scalable GP:")
                print(f"    Time: {scalable_time_mean[i]:.3f} ± {scalable_time_std[i]:.3f} s")
                print(f"    Mutual information: {scalable_info_gain_mean[i]:.3f} ± {scalable_info_gain_std[i]:.3f}")
                print(f"    Feasibility: {scalable_feasible_mean[i]:.1f} ± {scalable_feasible_std[i]:.1f} %")
            if not np.isnan(numpy_time_mean[i]) and not np.isnan(scalable_time_mean[i]):
                speedup = numpy_time_mean[i] / scalable_time_mean[i]
                print(f"    Speedup: {speedup:.2f}x")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate MPC exploration sweep results')
    parser.add_argument('--large-fonts', action='store_true',
                        help='Enable larger font sizes for subfigure display (default: use normal fonts)')
    parser.add_argument('--timestamp', type=str, default="20260213_static_10_random_seeds",
                        help='Timestamp of sweep results (default: 20260213_static_10_random_seeds)')
    args = parser.parse_args()
    
    use_large_fonts = args.large_fonts
    
    # Specify result directories
    timestamp = args.timestamp
    
    initial_samples_dir = f"experiments/thesis_results/initial_samples_sweep_{timestamp}"
    
    # Output directory for evaluation plots
    output_dir = f"experiments/paper_plots_dev/evaluation_plots_{timestamp}"
    
    # Check if directories exist
    if os.path.exists(initial_samples_dir):
        print("Evaluating initial samples sweep...")
        print(f"Using large fonts: {use_large_fonts}")
        plot_initial_samples_comparison(initial_samples_dir, output_dir, use_large_fonts)
    else:
        print(f"Initial samples results not found: {initial_samples_dir}")
        print("Please update the timestamp or run the sweep first.")


if __name__ == "__main__":
        main()
