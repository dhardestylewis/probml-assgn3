# -*- coding: utf-8 -*-
"""
probml - A modular package for probabilistic machine learning on tabular data.

Modules are imported lazily to avoid import errors during incremental extraction.
"""

__version__ = "0.1.0"

# Core utilities (always available)
from probml.core.utils import get_logger, check_available_methods
from probml.core.ica import ICAModule, PicardResult
