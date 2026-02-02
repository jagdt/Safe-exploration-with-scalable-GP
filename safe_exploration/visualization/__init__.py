from .plot_model_error import plot_model_error_comparison
from .plot_trajectory import (
    setup_trajectory_plot,
    add_trajectory_colorbar_and_legend,
    save_trajectory_plot,
)
from .styles import (
    # Colors
    RWTH_BLUE, RWTH_BLACK, RWTH_MAGENTA, RWTH_YELLOW,
    RWTH_PETROL, RWTH_TURQUOISE, RWTH_GREEN, RWTH_MAYGREEN,
    RWTH_ORANGE, RWTH_RED, RWTH_BORDEAUX, RWTH_PURPLE,
    RWTH_VIOLET, RWTH_LIGHT_BLUE, RWTH_GRAY,
    # Colormaps
    RWTH_CMAP, create_rwth_trajectory_colormap,
    # Configuration
    configure_matplotlib, PUBLICATION_RCPARAMS,
    # Utilities
    get_axis_label, get_linewidth, get_markersize,
    AXIS_LABELS,
)

__all__ = [
    'plot_model_error_comparison',
    # Trajectory plotting
    'setup_trajectory_plot',
    'add_trajectory_colorbar_and_legend',
    'save_trajectory_plot',
    # Colors
    'RWTH_BLUE', 'RWTH_BLACK', 'RWTH_MAGENTA', 'RWTH_YELLOW',
    'RWTH_PETROL', 'RWTH_TURQUOISE', 'RWTH_GREEN', 'RWTH_MAYGREEN',
    'RWTH_ORANGE', 'RWTH_RED', 'RWTH_BORDEAUX', 'RWTH_PURPLE',
    'RWTH_VIOLET', 'RWTH_LIGHT_BLUE', 'RWTH_GRAY',
    # Colormaps
    'RWTH_CMAP', 'create_rwth_trajectory_colormap',
    # Configuration
    'configure_matplotlib', 'PUBLICATION_RCPARAMS',
    # Utilities
    'get_axis_label', 'get_linewidth', 'get_markersize',
    'AXIS_LABELS',
]
