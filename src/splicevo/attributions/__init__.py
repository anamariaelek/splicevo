"""
Attribution computation and visualization for splice site and usage predictions.

This module provides functions to compute input gradient attributions (feature importance)
for model predictions, both for splice site classification and usage/condition-specific predictions.
"""

from .compute import (
    compute_attribution_splice,
    compute_attribution_usage,
    compute_attributions_for_sequence,
    compute_attributions_batch,
)

from .plot import (
    plot_attributions_splice,
    plot_attributions_usage,
    create_attribution_logo,
    extract_attribution_values,
    BASE_COLORS,
    BASE_NAMES,
)

__all__ = [
    # Computation
    'compute_attribution_splice',
    'compute_attribution_usage',
    'compute_attributions_for_sequence',
    'compute_attributions_batch',
    # Plotting
    'plot_attributions_splice',
    'plot_attributions_usage',
    'create_attribution_logo',
    'extract_attribution_values',
    'BASE_COLORS',
    'BASE_NAMES',
]
