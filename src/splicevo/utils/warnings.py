"""Utility to suppress common deprecation warnings from dependencies."""

import warnings
import functools

def suppress_pkg_resources_warnings(func):
    """Decorator to suppress pkg_resources deprecation warnings."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="pkg_resources is deprecated.*",
                category=UserWarning
            )
            return func(*args, **kwargs)
    return wrapper

def configure_warnings():
    """Configure warnings for the splicevo package."""
    # Suppress pkg_resources deprecation warning from norns/tangermeme
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated.*",
        category=UserWarning,
        module="norns.*"
    )

# Auto-configure when imported
configure_warnings()