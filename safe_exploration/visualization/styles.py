#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified visualization styles and configuration for safe exploration plots.

This module provides a single source of truth for:
- Matplotlib rcParams configuration
- RWTH Aachen University corporate colors
- Custom colormaps
- Plotting utilities and constants
"""

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _has_matplotlib = True
except ImportError:
    _has_matplotlib = False


# ============================================================================
# RWTH Aachen University Corporate Colors
# ============================================================================

RWTH_BLUE = '#00549F'
RWTH_BLACK = '#000000'
RWTH_MAGENTA = '#E30066'
RWTH_YELLOW = '#FFED00'
RWTH_PETROL = '#006165'
RWTH_TURQUOISE = '#0098A1'
RWTH_GREEN = '#57AB27'
RWTH_MAYGREEN = '#BDCD00'
RWTH_ORANGE = '#F6A800'
RWTH_RED = '#CC071E'
RWTH_BORDEAUX = '#A11035'
RWTH_PURPLE = '#612158'
RWTH_VIOLET = '#7A6FAC'
RWTH_LIGHT_BLUE = '#8EBAE5'
RWTH_GRAY = '#9C9E9F'


# ============================================================================
# Custom Colormaps
# ============================================================================

def create_rwth_trajectory_colormap(n_bins=256):
    """Create RWTH trajectory colormap: Light Blue -> Blue -> Magenta
    
    Parameters
    ----------
    n_bins : int
        Number of discrete colors in the colormap
        
    Returns
    -------
    LinearSegmentedColormap
        Custom RWTH trajectory colormap
    """
    colors_list = [RWTH_LIGHT_BLUE, RWTH_BLUE, RWTH_MAGENTA]
    return LinearSegmentedColormap.from_list('rwth_trajectory', colors_list, N=n_bins)


# Pre-create the default trajectory colormap
if _has_matplotlib:
    RWTH_CMAP = create_rwth_trajectory_colormap()
else:
    RWTH_CMAP = None


# ============================================================================
# Publication-Quality Plot Configuration
# ============================================================================

PUBLICATION_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 3.0,
    'lines.markersize': 6,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'figure.figsize': (8, 6),
    'axes.grid': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
}


def configure_matplotlib():
    """Configure matplotlib with publication-quality settings."""
    if not _has_matplotlib:
        return
    
    config = PUBLICATION_RCPARAMS.copy()
    plt.rcParams.update(config)


# ============================================================================
# Axis Label Mappings
# ============================================================================

# LaTeX-formatted axis labels for common variables
AXIS_LABELS = {
    'dθ': r'$\dot{\vartheta}$ [rad/s]',
    'θ': r'$\vartheta$ [rad]',
    'u': r'$u$ [Nm]',
    'Δ(dθ)': r'$\Delta \dot{\vartheta}$ [rad/s]',
    'Δ(θ)': r'$\Delta \vartheta$ [rad]',
    # Short forms without units (for titles/legends)
    'Δ(dθ)_short': r'$\Delta \dot{\vartheta}$',
    'Δ(θ)_short': r'$\Delta \vartheta$',
    # Full descriptive labels
    'angular_velocity': r'Angular velocity $\dot{\vartheta}$ [rad/s]',
    'angle': r'Angle $\vartheta$ [rad]',
}


def get_axis_label(key, default=None):
    """Get formatted axis label.
    
    Parameters
    ----------
    key : str
        Variable name key
    default : str, optional
        Default label if key not found
        
    Returns
    -------
    str
        Formatted axis label
    """
    return AXIS_LABELS.get(key, default or key)


# ============================================================================
# Utility Functions
# ============================================================================

def get_linewidth(default=2.5):
    """Get linewidth from rcParams or use default.
    
    Parameters
    ----------
    default : float
        Default linewidth if not in rcParams
        
    Returns
    -------
    float
        Linewidth value
    """
    if not _has_matplotlib:
        return default
    return plt.rcParams.get('lines.linewidth', default)


def get_markersize(default=6):
    """Get markersize from rcParams or use default.
    
    Parameters
    ----------
    default : float
        Default markersize if not in rcParams
        
    Returns
    -------
    float
        Markersize value
    """
    if not _has_matplotlib:
        return default
    return plt.rcParams.get('lines.markersize', default)


# ============================================================================
# Initialize on Import
# ============================================================================

# Automatically configure matplotlib when this module is imported
if _has_matplotlib:
    configure_matplotlib()
