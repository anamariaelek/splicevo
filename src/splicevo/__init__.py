"""Top-level package for splicevo."""

__author__ = """Anamaria Elek"""
__email__ = 'a.elek@zmbh.uni-heidelberg.de'

# Configure warnings before importing other modules
from .utils.warnings import configure_warnings
configure_warnings()

# Import main functionality
from . import data
from . import io

__all__ = ['data', 'io']
