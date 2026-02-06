#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trajectory plotting utilities for safe exploration visualization.

This module provides functions to create and update trajectory plots during
exploration, including safety bounds, domain bounds, initial samples, colorbars,
and legends.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .styles import (
    RWTH_BLACK, RWTH_PETROL, RWTH_GRAY, RWTH_LIGHT_BLUE,
    RWTH_MAGENTA, RWTH_ORANGE, RWTH_GREEN, RWTH_CMAP,
    get_linewidth, get_markersize, get_axis_label,
)


def setup_trajectory_plot(env, exploration_module, config, n_iterations):
    """Setup initial trajectory plot with safety bounds, domain bounds, and initial samples.
    
    Parameters
    ----------
    env : Environment
        The environment instance
    exploration_module : SafeMPCExploration
        The exploration module with trained model
    config : Config
        Configuration object with visualization settings
    n_iterations : int
        Total number of iterations (for colorbar)
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes
    """
    # Create base plot with safety bounds
    fig, ax = env.plot_safety_bounds(color=RWTH_BLACK, normalize=False)
    
    # Set axis labels and title
    ax.set_xlabel(get_axis_label('angular_velocity'))
    ax.set_ylabel(get_axis_label('angle'))
    
    # Plot domain bounds if available
    _plot_domain_bounds(ax, env, exploration_module)
    
    # Plot initial training samples if requested
    if config.visualize_initial_samples:
        _plot_initial_samples(ax, env, exploration_module)
    
    return fig, ax


def add_trajectory_colorbar_and_legend(fig, ax, config, exploration_module, 
                                        n_iterations, verify_safety):
    """Add colorbar and legend to trajectory plot.
    
    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes
    config : Config
        Configuration object with visualization settings
    exploration_module : SafeMPCExploration
        The exploration module
    n_iterations : int
        Total number of iterations
    verify_safety : bool
        Whether safety verification is enabled
        
    Returns
    -------
    None
    """
    if n_iterations <= 1:
        return
    
    # Add colorbar for iteration coloring
    sm = plt.cm.ScalarMappable(cmap=RWTH_CMAP, norm=plt.Normalize(vmin=1, vmax=n_iterations))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Exploration step', rotation=270, labelpad=20)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Check visualization flags
    show_ellipsoids = verify_safety and config.visualize_ellipsoids
    show_safe_traj = verify_safety and config.visualize_safe_trajectory
    
    # Determine if legend is needed
    has_legend_items = (
        config.visualize_initial_samples or 
        show_ellipsoids or 
        show_safe_traj or 
        _has_domain_bounds(exploration_module)
    )
    
    if has_legend_items:
        legend_elements = _create_legend_elements(
            config, exploration_module, verify_safety, show_ellipsoids, show_safe_traj
        )
        ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9)


def _plot_domain_bounds(ax, env, exploration_module):
    """Plot domain bounds on the trajectory plot.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    env : Environment
        The environment instance
    exploration_module : SafeMPCExploration
        The exploration module
    """
    if not hasattr(exploration_module.safempc.ssm, 'domain_lengths'):
        return
    
    domain_lengths = exploration_module.safempc.ssm.domain_lengths
    if domain_lengths is None:
        return
    
    domain_lengths = np.array(domain_lengths)
    
    # Create normalized domain bounds
    domain_bounds_norm = np.zeros((len(domain_lengths), 2))
    domain_bounds_norm[:, 0] = -domain_lengths / 2
    domain_bounds_norm[:, 1] = domain_lengths / 2
    
    if len(domain_bounds_norm) < env.n_s:
        return
    
    state_domain_bounds_norm = domain_bounds_norm[:env.n_s, :]
    state_norm_factors = env.norm[0]
    
    # Unnormalize to physical coordinates
    state_domain_bounds_phys = state_domain_bounds_norm * state_norm_factors[:, np.newaxis]
    
    # For 2D state space
    if env.n_s == 2:
        x_min, x_max = state_domain_bounds_phys[0, :]
        y_min, y_max = state_domain_bounds_phys[1, :]
        
        # Get current axis limits to define outer boundary
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Create polygon with hole (outer boundary with inner cutout)
        outer_x = [xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]]
        outer_y = [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]]
        inner_x = [x_min, x_min, x_max, x_max, x_min]
        inner_y = [y_min, y_max, y_max, y_min, y_min]
        
        verts = list(zip(outer_x + inner_x, outer_y + inner_y))
        codes = ([Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY] +
                 [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY])
        path = Path(verts, codes)
        
        # Fill outside region
        patch = PathPatch(path, facecolor=RWTH_PETROL, alpha=0.1, edgecolor='none')
        ax.add_patch(patch)
        
        # Draw domain bounds border
        linewidth = get_linewidth()
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=linewidth, edgecolor=RWTH_PETROL,
                        facecolor='none', label='Domain bounds')
        ax.add_patch(rect)


def _plot_initial_samples(ax, env, exploration_module):
    """Plot initial training samples on the trajectory plot.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    env : Environment
        The environment instance
    exploration_module : SafeMPCExploration
        The exploration module
    """
    x_train_init = exploration_module.x_train
    n_train, _ = np.shape(x_train_init)
    
    for i in range(n_train):
        ax = env.plot_state(ax, x_train_init[i, :env.n_s], 
                           color=RWTH_GRAY, normalize=False, unnormalize=True)


def _has_domain_bounds(exploration_module):
    """Check if domain bounds are available.
    
    Parameters
    ----------
    exploration_module : SafeMPCExploration
        The exploration module
        
    Returns
    -------
    bool
        True if domain bounds are available
    """
    if not hasattr(exploration_module.safempc.ssm, 'domain_lengths'):
        return False
    
    domain_lengths = exploration_module.safempc.ssm.domain_lengths
    if domain_lengths is None:
        return False
    
    # Check if valid for 2D
    domain_lengths = np.array(domain_lengths)
    return len(domain_lengths) >= 2


def _create_legend_elements(config, exploration_module, verify_safety, 
                           show_ellipsoids=True, show_safe_traj=True):
    """Create legend elements for the trajectory plot.
    
    Parameters
    ----------
    config : Config
        Configuration object with visualization settings
    exploration_module : SafeMPCExploration
        The exploration module
    verify_safety : bool
        Whether safety verification is enabled
    show_ellipsoids : bool
        Whether ellipsoids are shown (default: True)
    show_safe_traj : bool
        Whether safe trajectory is shown (default: True)
        
    Returns
    -------
    list
        List of Line2D legend elements
    """
    legend_elements = []
    markersize = get_markersize()
    linewidth = get_linewidth()
    
    # Initial samples legend entries
    if config.visualize_initial_samples:
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor=RWTH_GRAY,
                   markersize=markersize, alpha=0.3, linestyle='', label='Initial samples'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=RWTH_LIGHT_BLUE,
                   markersize=markersize, linestyle='', label='Early exploration'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=RWTH_MAGENTA,
                   markersize=markersize, linestyle='', label='Late exploration'),
        ])
    
    # Safe region boundary
    legend_elements.append(
        Line2D([0], [0], color=RWTH_BLACK, linewidth=linewidth,
               label='Safe region')
    )
    
    # Domain bounds legend entry
    if _has_domain_bounds(exploration_module):
        legend_elements.append(
            Line2D([0], [0], color=RWTH_PETROL, linewidth=linewidth,
                   label='Domain bounds')
        )
    
    # Propagated uncertainty (safety verification)
    if show_ellipsoids:
        legend_elements.append(
            Line2D([0], [0], color=RWTH_ORANGE, linewidth=linewidth,
                   label='Propagated uncertainty')
        )
    
    if show_safe_traj:
        legend_elements.append(
            Line2D([0], [0], color=RWTH_GREEN, linewidth=0, marker='o',
                   markersize=6, label='Safe trajectory')
        )
    
    return legend_elements


def save_trajectory_plot(fig, save_path, filename='trajectory_final.png'):
    """Save trajectory plot to file.
    
    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save
    save_path : str
        Directory path to save the plot
    filename : str
        Filename for the saved plot (default: 'trajectory_final.png')
        
    Returns
    -------
    str
        Full path to saved file
    """
    full_path = f"{save_path}/{filename}"
    fig.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved trajectory plot: {full_path}")
    return full_path
