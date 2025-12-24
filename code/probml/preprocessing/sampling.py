# -*- coding: utf-8 -*-
from probml.core.utils import get_logger

import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.neighbors import KernelDensity
from typing import Tuple, Optional, Dict, List, Union
import logging


logger = get_logger(__name__) # Module-level logger

class SamplingUtils:
    """
    Provides utility functions for data sampling, including downsampling,
    upsampling with optional KDE synthesis, and stratified sampling.
    Includes functionality to return sample weights after rebalancing.
    """
    def __init__(self, random_state: Optional[int] = None, verbose: bool = True):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state) # Compatible with older numpy
        self.verbose = verbose
        # Instance logger, configured by the verbose flag
        self.logger = get_logger(f"{self.__class__.__name__}", verbose=self.verbose)


    def _log(self, message: str, level: str = "info"):
        """Helper for conditional logging using the instance logger."""
        log_func = getattr(self.logger, level, self.logger.info)
        log_func(message)


    def downsample(self,
                   df: pd.DataFrame,
                   n_samples: int,
                   stratify_col: Optional[str] = None,
                   replace: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Downsamples a DataFrame, with optional stratification.

        Args:
            df: Input DataFrame.
            n_samples: Desired number of samples.
            stratify_col: Column for stratified sampling.
            replace: Sample with replacement (False for standard downsampling).

        Returns:
            Tuple: (downsampled DataFrame, indices of sampled rows from original df).
        """
        n_total = len(df)
        if n_samples >= n_total and not replace:
            self._log(f"Requested n_samples ({n_samples}) >= total ({n_total}) and replace=False. Returning original.")
            return df.copy(), df.index.values # Return original indices

        if n_samples <= 0:
            self._log("n_samples <= 0. Returning empty DataFrame.", level="warning")
            return pd.DataFrame(columns=df.columns), np.array([])

        sampled_indices: np.ndarray = np.array([], dtype=df.index.dtype) # Match index type

        if stratify_col and stratify_col in df.columns and df[stratify_col].nunique() > 1:
            self._log(f"Stratified downsampling by '{stratify_col}' to {n_samples} samples.")
            try:
                sampled_indices_list = []
                stratum_counts = df[stratify_col].value_counts()
                target_props = stratum_counts / n_total
                target_samples_float = target_props * n_samples
                target_samples_int = np.floor(target_samples_float).astype(int)
                remainder = n_samples - target_samples_int.sum()

                # Distribute remainder based on fractional parts
                fractional_parts = target_samples_float - target_samples_int
                # Get indices of strata to add remainder to (those with largest fractional parts)
                # These are indices *within* the `target_samples_int` Series, corresponding to stratum labels
                add_indices_order = fractional_parts.index[np.argsort(-fractional_parts.values)]

                for i in range(remainder):
                    stratum_to_increment = add_indices_order[i % len(add_indices_order)] # Cycle if remainder > num_strata
                    target_samples_int[stratum_to_increment] += 1

                total_sampled_count = 0
                for stratum_val, target_stratum_samples in target_samples_int.items():
                    group = df[df[stratify_col] == stratum_val]
                    stratum_n_total = len(group)
                    stratum_n_to_sample = min(target_stratum_samples, stratum_n_total)
                    total_sampled_count += stratum_n_to_sample
                    if stratum_n_to_sample > 0:
                        stratum_sampled_orig_indices = self.rng.choice(group.index.values, size=stratum_n_to_sample, replace=replace)
                        sampled_indices_list.extend(stratum_sampled_orig_indices)
                    elif target_stratum_samples > 0:
                         self._log(f"Stratum '{stratum_val}' has {stratum_n_total}, needed {target_stratum_samples}, sampling {stratum_n_to_sample}.", level="debug")

                sampled_indices = np.array(list(set(sampled_indices_list)), dtype=df.index.dtype)

                if len(sampled_indices) != n_samples:
                    self._log(f"Stratified sampling resulted in {len(sampled_indices)} unique samples (target {n_samples}). Adjusting randomly.", level="warning")
                    if len(sampled_indices) > n_samples:
                        sampled_indices = self.rng.choice(sampled_indices, size=n_samples, replace=False)
                    elif len(sampled_indices) < n_samples:
                        n_needed_more = n_samples - len(sampled_indices)
                        remaining_indices = np.setdiff1d(df.index.values, sampled_indices)
                        if len(remaining_indices) >= n_needed_more:
                            extra_indices = self.rng.choice(remaining_indices, size=n_needed_more, replace=False)
                            sampled_indices = np.concatenate([sampled_indices, extra_indices])
                        else: # Not enough remaining, take all that are left
                            sampled_indices = np.concatenate([sampled_indices, remaining_indices])
                            self._log(f"Could only add {len(remaining_indices)} more samples. Final count: {len(sampled_indices)}.", level="warning")
            except Exception as e:
                self._log(f"Stratified sampling failed: {e}. Falling back to random.", level="error")
                sampled_indices = self.rng.choice(df.index.values, size=n_samples, replace=replace)
        else:
            if stratify_col: self._log(f"Stratify col '{stratify_col}' invalid or has only one unique value. Using random downsampling.")
            self._log(f"Random downsampling to {n_samples} samples.")
            sampled_indices = self.rng.choice(df.index.values, size=n_samples, replace=replace)

        if replace and len(np.unique(sampled_indices)) < len(sampled_indices):
             self._log("Sampling with replacement generated duplicates. Taking unique.", level="debug")
             sampled_indices = np.unique(sampled_indices)
             if len(sampled_indices) < n_samples: self._log(f"Unique samples ({len(sampled_indices)}) < target after unique().", level="warning")

        if len(sampled_indices) > n_samples: # Ensure correct final size if unique() reduced count or other adjustments
             sampled_indices = sampled_indices[:n_samples]
        elif len(sampled_indices) < n_samples and not replace: # Try to fill if possible (only if not sampling with replacement)
            self._log(f"Final sample count {len(sampled_indices)} is less than target {n_samples} after downsampling. This can happen if unique strata are very small.", level="warning")


        df_sampled = df.loc[sampled_indices].copy()
        self._log(f"Downsampled from {n_total} to {len(df_sampled)} samples.")
        return df_sampled, sampled_indices


    def rebalance_price_bands(self,
                              df: pd.DataFrame,
                              price_band_col: str = 'price_band',
                              log_price_col: Optional[str] = None, # Column with log-transformed prices (e.g., log_sale_price)
                              price_col: Optional[str] = None, # Column with original scale prices (e.g., sale_price)
                              target_balance: Union[str, Dict[str, int]] = 'median',
                              upsample_method: str = 'kde', # 'kde' or 'bootstrap'
                              downsample_overrepresented: bool = True,
                              return_weights: bool = True
                             ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """
        Rebalances DataFrame based on price bands (specified in `price_band_col`),
        optionally returning sample weights.
        Weights are calculated for upsampled bands. Weight = 1.0 otherwise.

        The log_price_col (e.g., 'log_sale_price') and price_col (e.g., 'sale_price')
        are used specifically for KDE upsampling if that method is chosen.
        """
        # Ensure the input DataFrame has a simple numeric index for internal operations
        # This helps prevent TypeErrors if the original df had a non-numeric index
        # and any internal logic (even if not currently present) were to rely on numeric index properties.
        df_original_indexed = df.copy() # Keep a copy with original index if needed later (not currently used)
        df = df.reset_index(drop=True)

        if price_band_col not in df.columns:
            self.logger.error(f"Price band column '{price_band_col}' not found in DataFrame.")
            return df_original_indexed if not return_weights else (df_original_indexed, pd.Series(1.0, index=df_original_indexed.index))

        log_price_col_for_kde = log_price_col
        price_col_for_kde = price_col

        if upsample_method == 'kde':
            if not log_price_col_for_kde or not price_col_for_kde:
                self.logger.warning(f"KDE upsampling requires both 'log_price_col' (e.g., 'log_sale_price') and 'price_col' (e.g., 'sale_price') to be specified. "
                                   f"Provided: log_price_col='{log_price_col_for_kde}', price_col='{price_col_for_kde}'. Defaulting to 'bootstrap'.")
                upsample_method = 'bootstrap'
            elif log_price_col_for_kde not in df.columns or price_col_for_kde not in df.columns:
                self.logger.warning(f"KDE upsampling requires columns '{log_price_col_for_kde}' and '{price_col_for_kde}' to be present in the DataFrame. "
                                   f"Missing: "
                                   f"{[c for c in [log_price_col_for_kde, price_col_for_kde] if c not in df.columns]}. "
                                   f"Defaulting to 'bootstrap'.")
                upsample_method = 'bootstrap'

        self._log(f"--- Rebalancing by '{price_band_col}' (Upsample method: '{upsample_method}', "
                  f"Log price col for KDE: '{log_price_col_for_kde if upsample_method == 'kde' else 'N/A'}', "
                  f"Original price col for KDE: '{price_col_for_kde if upsample_method == 'kde' else 'N/A'}', "
                  f"Downsample overrepresented: {downsample_overrepresented}, Return weights: {return_weights}) ---", level="info")

        counts = df[price_band_col].value_counts().to_dict()
        self._log(f"Original counts for bands in '{price_band_col}': {counts}", level="info")

        final_target_counts: Dict[Union[str, int], int] = {} # Allow for numeric band labels
        target_val = 0
        if isinstance(target_balance, dict):
            final_target_counts = target_balance.copy()
            for band in counts:
                if band not in final_target_counts: final_target_counts[band] = counts[band]
        elif isinstance(target_balance, str):
            valid_counts = [c for c in counts.values() if c > 0]
            if not valid_counts:
                self.logger.error(f"No bands with samples found in column '{price_band_col}'. Cannot determine target value for rebalancing.");
                return df_original_indexed if not return_weights else (df_original_indexed, pd.Series(1.0, index=df_original_indexed.index))

            if target_balance == 'median': target_val = int(np.median(valid_counts))
            elif target_balance == 'mean': target_val = int(np.mean(valid_counts))
            elif target_balance == 'max': target_val = int(np.max(valid_counts))
            else:
                self.logger.error(f"Invalid target_balance string: '{target_balance}'. Using median.");
                target_val = int(np.median(valid_counts))
            target_val = max(1, target_val) # Ensure target is at least 1
            self._log(f"Base target value for each band: {target_val} (derived from '{target_balance}' of counts: {valid_counts})", level="info")
            for band in counts: final_target_counts[band] = target_val
        else:
            self.logger.error(f"Invalid target_balance type: {type(target_balance)}. Must be 'str' or 'dict'.");
            return df_original_indexed if not return_weights else (df_original_indexed, pd.Series(1.0, index=df_original_indexed.index))

        self._log(f"Effective target counts for '{price_band_col}': {final_target_counts}", level="info")

        resampled_dfs: List[pd.DataFrame] = []

        for band_label, current_count in counts.items():
            target_count_for_band = final_target_counts.get(band_label, current_count) # Default to current if somehow missing
            band_df = df[df[price_band_col] == band_label].copy() # Operate on copy with simple index

            if current_count == 0:
                self._log(f"Band '{band_label}' in '{price_band_col}' is empty. Skipping.", level="debug");
                continue

            if current_count < target_count_for_band: # Upsample
                n_needed = target_count_for_band - current_count
                self._log(f"Upsampling band '{band_label}' from {current_count} to {target_count_for_band} (+{n_needed}) using '{upsample_method}'. "
                          f"(Log price col: '{log_price_col_for_kde}', Orig price col: '{price_col_for_kde}')", level="info")
                resampled_dfs.append(band_df.copy()) # Add original samples first

                new_samples_df = pd.DataFrame()
                current_band_upsample_method = upsample_method # Allow fallback per band

                if current_band_upsample_method == 'kde' and current_count >= 5 and log_price_col_for_kde and price_col_for_kde:
                    try:
                        kde_prices = band_df[log_price_col_for_kde].dropna().values.reshape(-1, 1)
                        if len(kde_prices) < 2: raise ValueError(f"Not enough finite log prices in '{log_price_col_for_kde}' for KDE bandwidth (found {len(kde_prices)}).")

                        # Scott's rule for bandwidth, ensure it's positive
                        bandwidth = 1.06 * np.std(kde_prices) * (len(kde_prices) ** (-1./5))
                        bandwidth = max(bandwidth, 1e-4) # Ensure positive bandwidth

                        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(kde_prices)
                        new_log_prices_synthetic = kde.sample(n_needed, random_state=self.rng).flatten()
                        new_actual_prices_synthetic = np.expm1(new_log_prices_synthetic) # Transform back from log

                        # Bootstrap other features based on original samples in the band
                        # Original indices of band_df are 0 to len(band_df)-1 due to earlier reset_index
                        new_sample_indices_for_bootstrap = self.rng.choice(band_df.index, size=n_needed, replace=True)
                        new_samples_df = band_df.loc[new_sample_indices_for_bootstrap].copy()

                        new_samples_df[log_price_col_for_kde] = new_log_prices_synthetic
                        new_samples_df[price_col_for_kde] = new_actual_prices_synthetic
                        new_samples_df[price_band_col] = band_label # Ensure band label is correct
                        new_samples_df.reset_index(drop=True, inplace=True) # Ensure 0-based index for new samples

                    except Exception as e:
                        self.logger.warning(f"KDE upsampling failed for band '{band_label}' (using log_price_col='{log_price_col_for_kde}', price_col='{price_col_for_kde}'): {e}. Falling back to bootstrap.", exc_info=False)
                        current_band_upsample_method = 'bootstrap'
                        new_samples_df = pd.DataFrame() # Reset df if KDE failed partway

                if current_band_upsample_method == 'bootstrap':
                    if upsample_method == 'kde': # Log if this is a fallback
                         self._log(f"Using bootstrap for band '{band_label}' due to KDE failure or low count ({current_count} < 5).", level="debug")
                    # Resample from the original band_df (which has original data for this band)
                    new_samples_df = resample(band_df, replace=True, n_samples=n_needed, random_state=self.rng)
                    new_samples_df.reset_index(drop=True, inplace=True) # Ensure 0-based index

                if not new_samples_df.empty:
                    resampled_dfs.append(new_samples_df)
                elif n_needed > 0 : # Only warn if we actually needed samples
                    self.logger.warning(f"Upsampling for band '{band_label}' resulted in an empty DataFrame, though {n_needed} samples were needed.")

            elif current_count > target_count_for_band and downsample_overrepresented: # Downsample
                self._log(f"Downsampling band '{band_label}' from {current_count} to {target_count_for_band}.", level="info")
                # Use internal downsample method. band_df already has a simple 0-based index here.
                downsampled_band_df, _ = self.downsample(band_df, n_samples=target_count_for_band, replace=False)
                resampled_dfs.append(downsampled_band_df)
            else: # Keep as is (count matches target, or downsampling disabled)
                self._log(f"Keeping band '{band_label}' as is (count: {current_count}, target: {target_count_for_band}).", level="debug")
                resampled_dfs.append(band_df.copy())

        if not resampled_dfs:
            self.logger.warning("Rebalancing resulted in no data. Returning original DataFrame (with original index).")
            return df_original_indexed if not return_weights else (df_original_indexed, pd.Series(1.0, index=df_original_indexed.index))

        # Concatenate results - this creates a new default integer index (0 to N-1)
        df_rebalanced = pd.concat(resampled_dfs, ignore_index=True)

        final_weights_list = []
        current_pos = 0
        for temp_df_part in resampled_dfs: # Iterate through the same list of DataFrames used for concat
            num_samples_in_part = len(temp_df_part)
            if num_samples_in_part == 0: continue

            # Infer band label from the first row of this part (all rows in temp_df_part should have same band)
            # Use .mode()[0] for safety with categorical types or potential mixed types if data is messy.
            band_label_for_part = temp_df_part[price_band_col].mode()[0]
            original_count_for_part = counts.get(band_label_for_part, 0)
            target_count_for_part = final_target_counts.get(band_label_for_part, original_count_for_part)

            weight_for_part = 1.0
            if original_count_for_part > 0 and original_count_for_part < target_count_for_part : # Was upsampled
                weight_for_part = float(original_count_for_part) / target_count_for_part

            final_weights_list.extend([weight_for_part] * num_samples_in_part)
            current_pos += num_samples_in_part

        weights_series = pd.Series(final_weights_list, index=df_rebalanced.index, name='sample_weight')

        self._log(f"Rebalanced counts in '{price_band_col}': {df_rebalanced[price_band_col].value_counts().to_dict()}", level="info")
        self._log(f"Total samples after rebalancing: {len(df_rebalanced)}", level="info")

        if return_weights:
            if len(weights_series) != len(df_rebalanced):
                self.logger.error(f"Weight series length ({len(weights_series)}) mismatch with rebalanced df ({len(df_rebalanced)}). "
                                  "This indicates an issue in weight calculation logic. Returning default weights of 1.0.")
                weights_series = pd.Series(1.0, index=df_rebalanced.index, name='sample_weight')
            else:
                 self._log(f"Sample weights generated for rebalanced data (Min: {weights_series.min():.3f}, Max: {weights_series.max():.3f}, Mean: {weights_series.mean():.3f}).", level="info")
            return df_rebalanced, weights_series
        else:
            return df_rebalanced

