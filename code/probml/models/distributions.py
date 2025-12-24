# -*- coding: utf-8 -*-
from probml.core.utils import get_logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import (
    Normal, StudentT, Categorical, MixtureSameFamily, Independent,
    MultivariateNormal, Distribution,utils as dist_utils # For broadcast_all
)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split # Not used in this file directly, but common
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence, NamedTuple
import warnings
import time
import os
import logging
import math
from collections import defaultdict


# --- Custom Distributions (Potentially with enhancements) ---
class StudentTDistribution(StudentT):
    def __init__(self, df, loc, scale, validate_args=None, epsilon=1e-6):
        df = torch.clamp(df, min=epsilon)
        scale = torch.clamp(scale, min=epsilon)
        # Use torch.distributions.utils.broadcast_all for robust broadcasting
        try:
            self.df_broadcast, self.loc_broadcast, self.scale_broadcast = dist_utils.broadcast_all(df, loc, scale)
        except ValueError as e:
            raise ValueError(f"Broadcasting failed for StudentT params: df={df.shape}, loc={loc.shape}, scale={scale.shape}. Error: {e}") from e
        super().__init__(self.df_broadcast, self.loc_broadcast, self.scale_broadcast, validate_args=validate_args)
        self._epsilon = epsilon

    def rsample(self, sample_shape=torch.Size()):
        # Relying on parent's rsample which is correctly implemented
        if not hasattr(super(), 'rsample'):
            raise NotImplementedError("Parent StudentT class does not have rsample method.")
        return super().rsample(sample_shape)

class MixtureDistribution(MixtureSameFamily):
    def __init__(self, mixture_distribution: Categorical,
                 component_distribution: Distribution, validate_args=None):
        num_components = mixture_distribution.logits.shape[-1]
        current_logger_mix = get_logger(f"{self.__class__.__name__}", verbose=True) # Ensure logger is verbose for warnings

        if num_components == 0:
            current_logger_mix.warning("MixtureDistribution received 0 components. Defaulting to a single Normal component for stability.")
            # Create a dummy single component setup
            # Infer batch_shape and event_shape from the (empty) component_distribution
            dummy_event_shape = component_distribution.event_shape
            if isinstance(component_distribution, Independent): # For Independent distributions
                dummy_batch_shape = component_distribution.base_dist.batch_shape[:-1] # Exclude component dim
            else: # For distributions where last batch dim is component dim
                dummy_batch_shape = component_distribution.batch_shape[:-1] if len(component_distribution.batch_shape) > 0 else torch.Size([])

            device = mixture_distribution.logits.device
            dummy_mix_logits = torch.ones(dummy_batch_shape + torch.Size([1]), device=device) # Single component, prob 1
            mixture_distribution = Categorical(logits=dummy_mix_logits, validate_args=validate_args)

            # Create a default Normal component
            comp_loc = torch.zeros(dummy_batch_shape + torch.Size([1]) + dummy_event_shape, device=device)
            comp_scale = torch.ones(dummy_batch_shape + torch.Size([1]) + dummy_event_shape, device=device)
            base_dist = Normal(loc=comp_loc, scale=comp_scale, validate_args=validate_args)

            # If original component_distribution was Independent, wrap the new base_dist similarly
            if isinstance(component_distribution, Independent):
                 component_distribution = Independent(base_dist, len(dummy_event_shape)) # Reinterpret_batch_ndims should match original
            else:
                 component_distribution = base_dist # If original wasn't Independent

        super().__init__(mixture_distribution, component_distribution, validate_args=validate_args)

