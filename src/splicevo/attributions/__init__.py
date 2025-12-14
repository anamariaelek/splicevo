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
    compute_attributions_splice,
    compute_attributions_usage,
    save_attributions_for_modisco,
)

from .attributions import (
    AttributionCalculator,
)

from .plot import (
    plot_attributions_splice,
    plot_attributions_usage,
    plot_attributions_splice_from_result,
    plot_attributions_usage_from_result,
    create_attribution_logo,
    extract_attribution_values,
    BASE_COLORS,
    BASE_NAMES,
)

from .modisco_analysis import (
    ModiscoConfig,
    ModiscoInput,
    AttributionAggregator,
    ModiscoAnalyzer,
    analyze_attributions_quick,
)

__all__ = [
    # Computation - low-level
    'compute_attribution_splice',
    'compute_attribution_usage',
    'compute_attributions_for_sequence',
    'compute_attributions_batch',
    # Computation - high-level flexible API
    'compute_attributions_splice',
    'compute_attributions_usage',
    # Calculator class
    'AttributionCalculator',
    # Plotting - legacy format
    'plot_attributions_splice',
    'plot_attributions_usage',
    # Plotting - flexible API format (new)
    'plot_attributions_splice_from_result',
    'plot_attributions_usage_from_result',
    # Plotting utilities
    'create_attribution_logo',
    'extract_attribution_values',
    'save_attributions_for_modisco',
    'BASE_COLORS',
    'BASE_NAMES',
    # TF-MoDisco analysis - flexible API
    'ModiscoConfig',
    'ModiscoInput',
    'AttributionAggregator',
    'ModiscoAnalyzer',
    'analyze_attributions_quick',
]
