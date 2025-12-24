# -*- coding: utf-8 -*-
"""
Core utility functions for the probml package.
"""
import warnings
import time
import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# Optional imports checked at runtime
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from skdim.id import TwoNN  # type: ignore
    SKDIM_AVAILABLE = True
except ImportError:
    SKDIM_AVAILABLE = False

try:
    from picard import picard  # type: ignore
    PICARD_AVAILABLE = True
except ImportError:
    PICARD_AVAILABLE = False

try:
    import hdbscan  # type: ignore
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


def get_logger(name: str, verbose: bool = True) -> logging.Logger:
    """
    Create or get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        verbose: If True, set level to INFO; otherwise WARNING
        
    Returns:
        Configured Logger instance
    """
    logger_instance = logging.getLogger(name)
    if not logger_instance.handlers:  # Avoid duplicate handlers
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
    logger_instance.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger_instance


def check_available_methods(verbose: bool = True) -> dict:
    """
    Check which advanced methods are available and log their status.
    
    Args:
        verbose: Whether to print status messages
        
    Returns:
        Dictionary with availability status of each method
    """
    logger = get_logger(__name__, verbose)
    log_func = logger.info if verbose else lambda x: None
    
    log_func("\nChecking available advanced methods:")
    log_func(f"  FAISS (fast KNN): {'Available' if FAISS_AVAILABLE else 'Not available'}")
    log_func(f"  scikit-dim (TwoNN): {'Available' if SKDIM_AVAILABLE else 'Not available'}")
    log_func(f"  Picard (ICA): {'Available' if PICARD_AVAILABLE else 'Not available'}")
    log_func(f"  HDBSCAN: {'Available' if HDBSCAN_AVAILABLE else 'Not available'}")
    log_func("")
    
    return {
        "faiss": FAISS_AVAILABLE,
        "skdim": SKDIM_AVAILABLE,
        "picard": PICARD_AVAILABLE,
        "hdbscan": HDBSCAN_AVAILABLE,
    }


__all__ = [
    "get_logger",
    "check_available_methods",
    "FAISS_AVAILABLE",
    "SKDIM_AVAILABLE", 
    "PICARD_AVAILABLE",
    "HDBSCAN_AVAILABLE",
]
