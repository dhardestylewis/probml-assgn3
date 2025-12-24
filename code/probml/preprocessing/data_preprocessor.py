# -*- coding: utf-8 -*-
from probml.core.utils import get_logger
import pandas as pd
import numpy as np
import warnings
from typing import Tuple, List, Optional, Dict, Any, Union
import logging
from collections import defaultdict

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.ensemble import IsolationForest # For robust outlier detection
from scipy.stats import skew, kurtosis


class DataPreprocessor:
    """
    Handles preprocessing of real estate data, including cleaning, transformation,
    feature engineering (BBL), scaling, and robust outlier/percentile detection for prices.
    Retains all records, marking sale_price and log_sale_price as NaN for missing/non-positive original prices.
    """
    def __init__(self,
                 price_col: str = 'sale_price',
                 log_price_col_name: str = 'log_sale_price',
                 price_outlier_col_name: str = 'is_price_outlier', # Flag for detected outliers
                 outlier_method: str = 'isolation_forest', # Method for price outlier detection ('isolation_forest', 'percentile', 'both', 'none')
                 outlier_contamination: Union[str, float] = 'auto', # Contamination for IsolationForest
                 min_price_percentile_filter: Optional[float] = None, # e.g., 0.01 for 1st percentile (lower bound)
                 max_price_percentile_filter: Optional[float] = None, # e.g., 0.99 for 99th percentile (upper bound)
                 skewness_threshold: float = 1.0, # Threshold for auto-transforming other numeric features
                 default_transform_type: str = 'quantile', # Default transform for skewed features
                 random_state: Optional[int] = None,
                 verbose: bool = True):

        self.price_col = price_col
        self.log_price_col = log_price_col_name # This will be the name of the log-transformed price column
        self.price_outlier_col = price_outlier_col_name
        self.outlier_method = outlier_method.lower() if isinstance(outlier_method, str) else 'none'
        self.outlier_contamination = outlier_contamination

        self.min_price_percentile_filter = min_price_percentile_filter
        self.max_price_percentile_filter = max_price_percentile_filter
        self.fitted_min_price_cutoff_value: Optional[float] = None
        self.fitted_max_price_cutoff_value: Optional[float] = None
        self.fitted_min_percentile_threshold_used: Optional[float] = None
        self.fitted_max_percentile_threshold_used: Optional[float] = None

        self.skewness_threshold = abs(skewness_threshold)
        self.default_transform_type = default_transform_type.lower()
        self.random_state = random_state
        self.verbose = verbose
        self.logger = get_logger(f"{self.__class__.__name__}", verbose) # Instance logger

        self.feature_transformers: Dict[str, Any] = {}
        self.scaler: Optional[StandardScaler] = None
        self.isolation_forest_model: Optional[IsolationForest] = None

        self.transformed_features_list: List[str] = []
        self.scaled_features_list: List[str] = []
        self.feature_type_metadata: Dict[str, str] = {}
        self.price_analysis_stats: Dict[str, Any] = {}

        self.fitted = False

        valid_outlier_methods = ['isolation_forest', 'percentile', 'both', 'none']
        if self.outlier_method not in valid_outlier_methods:
            raise ValueError(f"Unsupported outlier_method: {self.outlier_method}. Supported: {valid_outlier_methods}.")

        self._log(f"DataPreprocessor initialized. Original price column: '{self.price_col}', "
                  f"Log-transformed price column to be created/used: '{self.log_price_col}', "
                  f"Skew Threshold: {self.skewness_threshold}, "
                  f"Default Feature Transform: '{self.default_transform_type}', Price Outlier Method: '{self.outlier_method}', "
                  f"Default Min Price Percentile: {self.min_price_percentile_filter}, "
                  f"Default Max Price Percentile: {self.max_price_percentile_filter})", level="info")

    def _log(self, message: str, level: str = "info", exc_info: bool = False):
        """Helper for conditional logging via the instance's logger."""
        # Ensure logger is correctly set for verbosity at the time of logging
        current_log_level = logging.DEBUG if level == "debug" else logging.INFO
        if not self.verbose and current_log_level < logging.WARNING: # Skip if not verbose and trying to log below WARNING
             pass
        else:
            log_func = getattr(self.logger, level, self.logger.info)
            log_func(message, exc_info=exc_info)


    def clean_and_transform_price(self,
                                  df: pd.DataFrame,
                                  min_price_pct_threshold_to_apply: Optional[float] = None,
                                  max_price_pct_threshold_to_apply: Optional[float] = None
                                 ) -> pd.DataFrame:
        """
        Cleans the price column (specified by `self.price_col`),
        creates/applies log1p to positive prices into `self.log_price_col`,
        and flags/filters outliers on positive prices.
        Rows with initially missing or non-positive prices will have NaN in `self.price_col` (if coerced)
        and in `self.log_price_col`. These rows are NOT dropped.
        """
        self._log(f"Starting price processing. Original price column: '{self.price_col}', "
                  f"Target log-price column: '{self.log_price_col}'. Initial shape: {df.shape}", level="info")
        _df = df.copy()

        if self.price_col not in _df.columns:
            self._log(f"Original price column '{self.price_col}' not found in DataFrame. Cannot process prices.", level="error")
            # Ensure log_price_col is created as all NaNs if price_col is missing, so downstream steps don't break.
            if self.log_price_col not in _df.columns:
                 _df[self.log_price_col] = np.nan
                 self._log(f"Created empty (all NaN) log-price column '{self.log_price_col}' as original price column was missing.", level="warning")
            return _df

        # Coerce to numeric. Non-numeric prices become NaN. Original NaNs remain NaN.
        original_nan_count = _df[self.price_col].isnull().sum()
        _df[self.price_col] = pd.to_numeric(_df[self.price_col], errors='coerce')
        coerced_nan_count = _df[self.price_col].isnull().sum()
        newly_coerced_nans = coerced_nan_count - original_nan_count
        if newly_coerced_nans > 0:
            self._log(f"{newly_coerced_nans} entries in '{self.price_col}' were coerced to NaN (non-numeric).", level="info")
        self._log(f"Total {coerced_nan_count} NaN values in '{self.price_col}' after numeric coercion.", level="info")

        # Initialize log_price_col. It will remain NaN for rows where price_col is NaN or non-positive.
        # This ensures the column self.log_price_col is always created.
        _df[self.log_price_col] = np.nan
        self._log(f"Initialized column '{self.log_price_col}' with NaNs. It will be populated with log1p of positive values from '{self.price_col}'.", level="debug")


        # Initialize outlier column if any price processing is expected
        if self.outlier_method != 'none' or \
           self.fitted_min_price_cutoff_value is not None or \
           min_price_pct_threshold_to_apply is not None or \
           self.fitted_max_price_cutoff_value is not None or \
           max_price_pct_threshold_to_apply is not None:
            _df[self.price_outlier_col] = False # Default: not an outlier

        # Identify rows with valid, positive prices for transformation and outlier detection
        positive_price_mask = (_df[self.price_col].fillna(-1) > 0) # Fill NaN with a non-positive value for the mask
        n_positive = positive_price_mask.sum()
        n_total_rows = len(_df)
        n_missing_or_non_positive = n_total_rows - n_positive
        self._log(f"Found {n_positive} positive prices in '{self.price_col}' (out of {n_total_rows} total rows). "
                  f"{n_missing_or_non_positive} rows have missing, zero, or negative prices in '{self.price_col}'.", level="info")

        current_min_cutoff = self.fitted_min_price_cutoff_value
        current_max_cutoff = self.fitted_max_price_cutoff_value

        # Percentile cutoffs are determined *only* from positive prices in self.price_col
        if n_positive > 0:
            if min_price_pct_threshold_to_apply is not None:
                current_min_cutoff = np.percentile(_df.loc[positive_price_mask, self.price_col], min_price_pct_threshold_to_apply * 100)
                self.fitted_min_price_cutoff_value = current_min_cutoff
                self.fitted_min_percentile_threshold_used = min_price_pct_threshold_to_apply
                self.price_analysis_stats['fitted_min_price_cutoff_value'] = current_min_cutoff
                self.price_analysis_stats['fitted_min_percentile_threshold_used'] = min_price_pct_threshold_to_apply
                self._log(f"Calculated min price cutoff for '{self.price_col}' at {min_price_pct_threshold_to_apply*100:.1f}th percentile of positive prices: {current_min_cutoff}", level="info")

            if max_price_pct_threshold_to_apply is not None:
                current_max_cutoff = np.percentile(_df.loc[positive_price_mask, self.price_col], max_price_pct_threshold_to_apply * 100)
                self.fitted_max_price_cutoff_value = current_max_cutoff
                self.fitted_max_percentile_threshold_used = max_price_pct_threshold_to_apply
                self.price_analysis_stats['fitted_max_price_cutoff_value'] = current_max_cutoff
                self.price_analysis_stats['fitted_max_percentile_threshold_used'] = max_price_pct_threshold_to_apply
                self._log(f"Calculated max price cutoff for '{self.price_col}' at {max_price_pct_threshold_to_apply*100:.1f}th percentile of positive prices: {current_max_cutoff}", level="info")

        percentile_filtered_indices = pd.Series(False, index=_df.index)
        if current_min_cutoff is not None:
            percentile_filtered_indices |= (_df[self.price_col] < current_min_cutoff) & positive_price_mask
        if current_max_cutoff is not None:
            percentile_filtered_indices |= (_df[self.price_col] > current_max_cutoff) & positive_price_mask

        if ('percentile' in self.outlier_method or 'both' in self.outlier_method) and self.price_outlier_col in _df.columns:
            if percentile_filtered_indices.any():
                _df.loc[percentile_filtered_indices, self.price_outlier_col] = True
                self._log(f"Flagged {percentile_filtered_indices.sum()} positive prices in '{self.price_col}' as outliers based on percentile cutoffs.", level="info")

        # Apply log1p transform to positive prices, storing in self.log_price_col
        if n_positive > 0:
            _df.loc[positive_price_mask, self.log_price_col] = np.log1p(_df.loc[positive_price_mask, self.price_col])
            self._log(f"Applied log1p transform to {n_positive} positive values from '{self.price_col}' and stored in '{self.log_price_col}'. "
                      f"Non-positive or NaN original prices in '{self.price_col}' result in NaN in '{self.log_price_col}'.", level="debug")

            # Isolation Forest outlier detection on log-transformed positive prices (from self.log_price_col)
            if ('isolation_forest' in self.outlier_method or 'both' in self.outlier_method) and self.price_outlier_col in _df.columns:
                log_positive_prices = _df.loc[positive_price_mask, self.log_price_col].values.reshape(-1, 1)
                finite_log_mask = np.isfinite(log_positive_prices.flatten())

                if not np.any(finite_log_mask):
                    self._log(f"No finite log-positive prices in '{self.log_price_col}' available for IsolationForest.", level="warning")
                    if min_price_pct_threshold_to_apply is not None: # i.e. during fit
                        self.isolation_forest_model = None
                else:
                    log_positive_prices_finite = log_positive_prices[finite_log_mask]
                    if log_positive_prices_finite.shape[0] < 2:
                        self._log(f"Too few finite positive log-price samples ({log_positive_prices_finite.shape[0]}) in '{self.log_price_col}' for IsolationForest. Skipping.", level="warning")
                        if min_price_pct_threshold_to_apply is not None:
                            self.isolation_forest_model = None
                    else:
                        try:
                            if min_price_pct_threshold_to_apply is not None or self.isolation_forest_model is None:
                                self._log(f"Fitting IsolationForest (contamination='{self.outlier_contamination}') on {log_positive_prices_finite.shape[0]} finite values from '{self.log_price_col}'.", level="debug")
                                if_model = IsolationForest(contamination=self.outlier_contamination, random_state=self.random_state, n_estimators=100)
                                if_model.fit(log_positive_prices_finite)
                                self.isolation_forest_model = if_model

                            if self.isolation_forest_model:
                                outlier_preds_finite = self.isolation_forest_model.predict(log_positive_prices_finite)
                                temp_if_outlier_flags = pd.Series(False, index=_df.loc[positive_price_mask].index)
                                indices_finite = _df.loc[positive_price_mask].index[finite_log_mask]
                                temp_if_outlier_flags.loc[indices_finite] = (outlier_preds_finite == -1)

                                if 'both' in self.outlier_method:
                                    _df.loc[positive_price_mask, self.price_outlier_col] |= temp_if_outlier_flags
                                else: # Overwrite/set with IF flags
                                    _df.loc[positive_price_mask, self.price_outlier_col] = temp_if_outlier_flags

                                num_if_outliers = temp_if_outlier_flags.sum()
                                self._log(f"IsolationForest identified {num_if_outliers} potential price outliers using '{self.log_price_col}'. Updated '{self.price_outlier_col}'.", level="info")
                            else:
                                self._log("Isolation Forest model not available for prediction.", level="warning")
                        except Exception as e:
                            self._log(f"Error during IsolationForest on '{self.log_price_col}': {e}", level="error", exc_info=True)
                            if min_price_pct_threshold_to_apply is not None:
                                self.isolation_forest_model = None
        else: # n_positive == 0
            self._log(f"No positive prices found in '{self.price_col}'. '{self.log_price_col}' will contain all NaNs. No outlier detection on prices performed.", level="info")

        try:
            valid_original_prices = _df[self.price_col].dropna()
            if not valid_original_prices.empty: self.price_analysis_stats['original_skew'] = skew(valid_original_prices)

            finite_log_prices = _df[self.log_price_col].replace([np.inf, -np.inf], np.nan).dropna()
            if not finite_log_prices.empty:
                self.price_analysis_stats['log_skew'] = skew(finite_log_prices)
                self.price_analysis_stats['log_kurtosis'] = kurtosis(finite_log_prices)
                self.price_analysis_stats['log_heavy_tailed'] = self.price_analysis_stats.get('log_kurtosis', 0) > 1.0
                self._log(f"Stats for '{self.price_col}': OrigSkew={self.price_analysis_stats.get('original_skew', 'N/A'):.2f}. "
                          f"Stats for '{self.log_price_col}': LogSkew={self.price_analysis_stats.get('log_skew', 'N/A'):.2f}, "
                          f"LogKurt={self.price_analysis_stats.get('log_kurtosis', 'N/A'):.2f} "
                          f"(HeavyTailed: {self.price_analysis_stats.get('log_heavy_tailed', False)})", level="debug")
            else:
                self._log(f"No finite values in '{self.log_price_col}' to calculate log skew/kurtosis.", level="debug")
                self.price_analysis_stats.update({'log_skew': np.nan, 'log_kurtosis': np.nan, 'log_heavy_tailed': False})
        except Exception as stat_e:
            self._log(f"Could not calculate price skew/kurtosis for '{self.price_col}' or '{self.log_price_col}': {stat_e}", level="warning")

        self.feature_type_metadata[self.price_col] = 'numeric_price_original'
        self.feature_type_metadata[self.log_price_col] = 'numeric_price_log_transformed_or_nan'
        if self.price_outlier_col in _df.columns:
            self.feature_type_metadata[self.price_outlier_col] = 'boolean_price_outlier_flag'

        # Final check on the log_price_col status
        if self.log_price_col in _df.columns:
            num_log_price_final_nan = _df[self.log_price_col].isnull().sum()
            self._log(f"Price processing complete for original price column '{self.price_col}'. "
                      f"Log-transformed price column '{self.log_price_col}' created/updated. "
                      f"'{self.log_price_col}' contains {num_log_price_final_nan} NaNs out of {len(_df)} rows. "
                      f"Shape after: {_df.shape}", level="info")
        else:
            self._log(f"Price processing attempted, but log-price column '{self.log_price_col}' is unexpectedly missing. "
                      f"Shape after: {_df.shape}", level="error")

        return _df

    def create_bbl_identifier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a standard BBL (Borough-Block-Lot) identifier string."""
        self._log("Attempting to create BBL identifier...", level="debug")
        _df = df.copy()
        bbl_components = ['borough', 'block', 'lot']

        if 'bbl' in _df.columns and _df['bbl'].notna().all(): # Check if 'bbl' already exists and is reasonably populated
            self._log("BBL column already exists and seems populated. Skipping creation.", level="debug")
            if 'bbl' not in self.feature_type_metadata: # Ensure metadata is set
                self.feature_type_metadata['bbl'] = 'identifier_existing'
            return _df

        missing_bbl_cols = [col for col in bbl_components if col not in _df.columns]
        if missing_bbl_cols:
            self._log(f"BBL component columns missing: {missing_bbl_cols}. Cannot create BBL identifier.", level="warning")
            return _df

        try:
            # Convert components to string, handling NaNs by treating them as invalid (-1 before str conversion)
            _df['borough_str'] = _df['borough'].fillna(-1).astype(float).astype(int).astype(str).str.strip()
            _df['block_str']   = _df['block'].fillna(-1).astype(float).astype(int).astype(str).str.strip().str.zfill(5)
            _df['lot_str']     = _df['lot'].fillna(-1).astype(float).astype(int).astype(str).str.strip().str.zfill(4)

            # Create BBL only where all components were valid (not resulting in '-1' based strings)
            # and do not contain hyphens (which might indicate parsing issues or actual negative numbers)
            valid_bbl_mask = ~(_df['borough_str'].isin(['-1', 'nan'])) & \
                             ~(_df['block_str'].isin([('-1').zfill(5), ('nan').zfill(5)])) & \
                             ~(_df['lot_str'].isin([('-1').zfill(4), ('nan').zfill(4)])) & \
                             ~(_df['borough_str'].str.contains('-')) & \
                             ~(_df['block_str'].str.contains('-')) & \
                             ~(_df['lot_str'].str.contains('-'))


            _df['bbl'] = pd.NA # Initialize with pandas NA for string type
            _df.loc[valid_bbl_mask, 'bbl'] = _df['borough_str'] + _df['block_str'] + _df['lot_str']

            num_created = valid_bbl_mask.sum()
            num_failed = len(_df) - num_created
            self._log(f"Created {num_created} BBL identifiers. {num_failed} rows had invalid/missing BBL components.", level="info")

            _df.drop(columns=['borough_str', 'block_str', 'lot_str'], inplace=True)
            self.feature_type_metadata['bbl'] = 'identifier_created'
        except Exception as e:
            self._log(f"Error creating BBL identifier: {e}", level="error", exc_info=True)
            if 'bbl' not in _df.columns: _df['bbl'] = pd.NA # Ensure column exists even if failed
        return _df

    def apply_feature_transformations(self,
                                      df: pd.DataFrame,
                                      numeric_cols: List[str],
                                      skewness_threshold_to_use: float,
                                      default_transform_type_to_use: str,
                                      auto_transform_skewed: bool = True
                                     ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        _df = df.copy()
        applied_transformers: Dict[str, Any] = {}
        transformed_cols_run: List[str] = []

        valid_numeric_cols = [col for col in numeric_cols if col in _df.columns and pd.api.types.is_numeric_dtype(_df[col])]
        if not valid_numeric_cols:
            self._log("No valid numeric columns provided for transformation.", level="warning")
            return _df, applied_transformers

        cols_to_transform_map: Dict[str, str] = {}
        effective_transform_type = default_transform_type_to_use.lower()
        effective_skew_threshold = abs(skewness_threshold_to_use)

        if auto_transform_skewed:
            self._log(f"Auto-detecting skewed features (Skew threshold > {effective_skew_threshold}). Applying '{effective_transform_type}'.", level="info")
            for col in valid_numeric_cols:
                col_data = _df[col].dropna()
                if len(col_data) > 2: # Skew calculation needs at least 3 non-NaN points
                    try:
                        col_skew = skew(col_data)
                        if abs(col_skew) > effective_skew_threshold:
                            cols_to_transform_map[col] = effective_transform_type
                    except Exception as e: self._log(f"Skew calculation failed for '{col}': {e}", level="warning")
            self._log(f"Found {len(cols_to_transform_map)} skewed columns for auto-transformation: {list(cols_to_transform_map.keys())}", level="info")
        elif effective_transform_type != 'none': # Apply to all if not auto and type is not none
            self._log(f"Applying '{effective_transform_type}' to all {len(valid_numeric_cols)} provided numeric columns (auto_transform_skewed=False).", level="info")
            for col in valid_numeric_cols: cols_to_transform_map[col] = effective_transform_type

        # Group columns by transformation type to apply batch transformations if possible
        grouped_cols: Dict[str, List[str]] = defaultdict(list)
        for col, t_type in cols_to_transform_map.items(): grouped_cols[t_type].append(col)

        for t_type, cols in grouped_cols.items():
            if not cols: continue
            self._log(f"Applying '{t_type}' transform to columns: {cols}", level="debug")
            X_subset = _df[cols].copy() # Operate on a copy for this transformation group

            # Impute NaNs before transformation
            if X_subset.isnull().any().any():
                self._log(f"NaNs detected before '{t_type}' transform for {cols}. Imputing with median.", level="warning")
                for c_name in cols: # Impute each column in the subset individually
                    if X_subset[c_name].isnull().any():
                        median_val = X_subset[c_name].median()
                        X_subset[c_name].fillna(median_val if not pd.isna(median_val) else 0.0, inplace=True)

            # Replace Infs after imputation
            if not np.all(np.isfinite(X_subset.values)):
                self._log(f"Non-finite values (Inf) detected after imputation for '{t_type}' transform for {cols}. Replacing with 0.", level="warning")
                for c_name in cols: # Replace in each column of the subset
                    X_subset[c_name].replace([np.inf, -np.inf], 0, inplace=True)

            transformer_instance = None; transformed_data = None
            # Unique key for the transformer instance based on type and columns it applies to
            # Useful if different groups of columns get the same type but fitted differently (though current logic fits per group)
            transformer_key = f"{t_type}_transformer_for_{'_'.join(sorted(cols))}"

            try:
                if t_type == 'quantile':
                    n_samples_subset = X_subset.shape[0]
                    # Adjust n_quantiles: must be <= n_samples. If n_samples is small, use n_samples-1 or a small default.
                    n_q = min(1000, max(10, n_samples_subset -1 if n_samples_subset > 1 else 1))
                    if n_q >= n_samples_subset and n_samples_subset > 0 : n_q = n_samples_subset -1 # strictly less
                    if n_q <=0 : n_q = min(10, n_samples_subset if n_samples_subset > 0 else 1) # smallest possible if very few samples

                    transformer_instance = QuantileTransformer(output_distribution='normal', n_quantiles=n_q, random_state=self.random_state, subsample=min(100_000, n_samples_subset))
                elif t_type == 'yeo-johnson':
                    transformer_instance = PowerTransformer(method='yeo-johnson', standardize=False) # standardize=False as we scale later
                elif t_type == 'box-cox':
                    # Box-Cox requires all data to be positive.
                    if np.any(X_subset.values <= 0):
                        self._log(f"Clipping non-positive values to a small positive number (e.g., 1e-6) for Box-Cox on {cols}.", level="warning")
                        X_subset = X_subset.clip(lower=1e-6)
                    transformer_instance = PowerTransformer(method='box-cox', standardize=False)
                elif t_type == 'log1p': # Direct numpy transform, not a sklearn transformer
                    if np.any(X_subset.values < 0): self._log(f"log1p encountered negative values in {cols}. Resulting NaNs will be cleaned post-transform.", level="warning")
                    transformed_data = np.log1p(X_subset.values)
                    # Clean NaNs/Infs that might result from log1p (e.g., log1p(-1) = -inf for complex, or from prior Infs)
                    transformed_data = np.nan_to_num(transformed_data, nan=0.0, posinf=0.0, neginf=0.0) # Replace with 0
                else:
                    self._log(f"Unsupported transform_type: '{t_type}'. Skipping for columns: {cols}.", level="warning"); continue

                if transformer_instance: # If a sklearn transformer was instantiated
                    transformed_data = transformer_instance.fit_transform(X_subset)
                    # Store the fitted transformer instance
                    self.feature_transformers[transformer_key] = transformer_instance

                if transformed_data is not None:
                    _df[cols] = transformed_data # Update original DataFrame subset
                    transformed_cols_run.extend(cols)
                    for col_name in cols: self.feature_type_metadata[col_name] = f'numeric_transformed_{t_type}'
            except Exception as e:
                self._log(f"Error during '{t_type}' transform for columns {cols}: {e}", level="error", exc_info=True)

        self.transformed_features_list = list(set(self.transformed_features_list + transformed_cols_run))
        # Ensure all numeric cols processed get some metadata type
        for col in valid_numeric_cols:
            if col not in transformed_cols_run and col not in self.feature_type_metadata: # If not transformed and no prior type
                self.feature_type_metadata[col] = 'numeric_original_untransformed' # More specific
        return _df, self.feature_transformers # Return transformers for `transform` method

    def scale_features(self, df: pd.DataFrame, features_to_scale: List[str]) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
        self._log(f"Applying StandardScaler to {len(features_to_scale)} features: {features_to_scale[:5]}...", level="info")
        _df = df.copy()
        valid_features = [f for f in features_to_scale if f in _df.columns and pd.api.types.is_numeric_dtype(_df[f])]
        if not valid_features:
            self._log("No valid numeric features found for scaling.", level="warning"); return _df, None

        X_to_scale = _df[valid_features].copy() # Operate on a copy
        # Impute NaNs before scaling
        if X_to_scale.isnull().any().any():
            cols_with_nan = X_to_scale.columns[X_to_scale.isnull().any()].tolist()
            self._log(f"NaNs detected before scaling columns: {cols_with_nan}. Imputing with median.", level="warning")
            for col in valid_features: # Impute each column individually
                if X_to_scale[col].isnull().any():
                    median_val = X_to_scale[col].median()
                    X_to_scale[col].fillna(median_val if not pd.isna(median_val) else 0.0, inplace=True)

        # Replace Infs after imputation
        if not np.all(np.isfinite(X_to_scale.values)):
            self._log("Non-finite values (Inf) detected before scaling after NaN imputation. Replacing with 0.", level="error")
            X_to_scale.replace([np.inf, -np.inf], 0, inplace=True) # Replace in the subset

        # Check for near-constant columns before scaling
        variances = X_to_scale.var()
        constant_cols = variances[variances < 1e-10].index.tolist()
        if constant_cols:
            self._log(f"Near-constant columns detected before scaling: {constant_cols}. Their variance is < 1e-10. StandardScaler will result in zeros for these.", level="warning")

        current_scaler = StandardScaler()
        try:
            scaled_values = current_scaler.fit_transform(X_to_scale)
            # Handle NaNs that might result from scaling constant columns (variance is 0)
            if np.isnan(scaled_values).any():
                nan_cols_after_scale = X_to_scale.columns[np.isnan(scaled_values).any(axis=0)].tolist()
                self._log(f"NaNs generated by StandardScaler for columns (likely due to zero variance): {nan_cols_after_scale}. Replacing these NaNs with 0.", level="warning")
                scaled_values = np.nan_to_num(scaled_values, nan=0.0) # Replace NaNs with 0

            _df[valid_features] = scaled_values # Update original DataFrame
            self.scaler = current_scaler # Store the fitted scaler
            self.scaled_features_list = list(set(self.scaled_features_list + valid_features))
            self._log(f"StandardScaler applied to {len(valid_features)} features.", level="info")
            for col in valid_features: # Update metadata
                current_type = self.feature_type_metadata.get(col, 'numeric') # Default if no prior transform type
                if '_scaled' not in current_type: self.feature_type_metadata[col] = current_type + '_scaled'
            return _df, self.scaler
        except Exception as e:
            self._log(f"Error scaling features {valid_features}: {e}", level="error", exc_info=True)
            return df.copy(), None # Return original df on error

    def fit_transform(self, df: pd.DataFrame,
                      numeric_features_to_process: Optional[List[str]] = None,
                      auto_transform_skewed: bool = True,
                      scale_numeric: bool = True,
                      create_bbl: bool = True,
                      min_price_percentile: Optional[float] = None,
                      max_price_percentile: Optional[float] = None,
                      default_transform_type_override: Optional[str] = None,
                      transform_skew_threshold_override: Optional[float] = None,
                      **kwargs # Allow extra kwargs
                     ) -> Optional[pd.DataFrame]:
        self.logger.info(f"--- Starting Preprocessing Pipeline (fit_transform) on DataFrame with shape {df.shape} ---")
        if kwargs:
            self._log(f"Received unexpected keyword arguments in fit_transform: {kwargs}. These will be ignored.", level="warning")

        # Reset all fitted states
        self.fitted = False; self.feature_transformers = {}; self.transformed_features_list = []
        self.scaler = None; self.scaled_features_list = []; self.feature_type_metadata = {}
        self.isolation_forest_model = None; self.price_analysis_stats = {}
        self.fitted_min_price_cutoff_value = None; self.fitted_max_price_cutoff_value = None
        self.fitted_min_percentile_threshold_used = None; self.fitted_max_percentile_threshold_used = None

        # Determine effective parameters for this run
        eff_min_price_pct_thresh = min_price_percentile if min_price_percentile is not None else self.min_price_percentile_filter
        eff_max_price_pct_thresh = max_price_percentile if max_price_percentile is not None else self.max_price_percentile_filter
        eff_skew_thresh = transform_skew_threshold_override if transform_skew_threshold_override is not None else self.skewness_threshold
        eff_transform_type = default_transform_type_override if default_transform_type_override is not None else self.default_transform_type

        self._log(f"Effective parameters for fit_transform: MinPricePct={eff_min_price_pct_thresh}, MaxPricePct={eff_max_price_pct_thresh}, "
                  f"PriceCol='{self.price_col}', LogPriceCol='{self.log_price_col}', "
                  f"SkewThreshold={eff_skew_thresh}, DefaultTransformType='{eff_transform_type}'", level="debug")

        _df = self.clean_and_transform_price(df.copy(), # Operate on a copy
                                             min_price_pct_threshold_to_apply=eff_min_price_pct_thresh,
                                             max_price_pct_threshold_to_apply=eff_max_price_pct_thresh)
        if _df is None or _df.empty:
            self.logger.error("DataFrame empty after price cleaning. Aborting fit_transform."); return None

        if create_bbl: _df = self.create_bbl_identifier(_df)

        # Define columns to exclude from numeric_features_to_process
        price_related_cols = [self.price_col, self.log_price_col, self.price_outlier_col]
        # Define a more comprehensive list of potential ID columns to exclude from general numeric processing
        potential_id_cols = [
            'bbl', 'borough', 'block', 'lot', 'zip_code', 'zipcode', # Common BBL and zip
            'cd', 'council', 'schooldist', 'healtharea', 'policeprct', # Admin/district IDs
            'xcoord', 'ycoord', 'latitude', 'longitude', # Coordinates
            'index', 'id', 'rowid', 'geoid', 'gid', 'uniqueid', 'parcelid', 'parid', 'parcelnumb', # Generic ID names
            'condono', 'appbbl', 'tbl', 'year', 'sale_year', 'sale_month', 'sale_day' # Other common IDs/time parts
        ]
        # Filter this list to only include those present in the DataFrame, case-insensitively
        df_cols_lower_map = {c.lower(): c for c in _df.columns}
        id_cols_in_df_cased = [df_cols_lower_map[id_col_lower] for id_col_lower in potential_id_cols if id_col_lower in df_cols_lower_map]

        cols_to_exclude_from_num_processing = list(set(price_related_cols + id_cols_in_df_cased))


        if numeric_features_to_process is None: # Auto-detect if not provided
            process_cols = [col for col in _df.select_dtypes(include=np.number).columns
                            if col not in cols_to_exclude_from_num_processing]
            self._log(f"Auto-detected {len(process_cols)} numeric features for transformation/scaling: {process_cols[:10]}...", level="debug")
        else: # Use provided list, but filter out non-numeric, price/ID cols
            process_cols = [f for f in numeric_features_to_process
                            if f in _df.columns and pd.api.types.is_numeric_dtype(_df[f])
                            and f not in cols_to_exclude_from_num_processing]
            self._log(f"Using specified list: {len(process_cols)} valid numeric features for transformation/scaling: {process_cols[:10]}...", level="debug")

        if process_cols:
            _df, fitted_transformers = self.apply_feature_transformations(
                _df, process_cols,
                skewness_threshold_to_use=eff_skew_thresh,
                default_transform_type_to_use=eff_transform_type,
                auto_transform_skewed=auto_transform_skewed
            )
            self.feature_transformers = fitted_transformers # Store fitted transformers
        else: self._log("No numeric features to apply transformations to (after exclusions).", level="info")

        if scale_numeric and process_cols:
            _df, fitted_scaler = self.scale_features(_df, features_to_scale=process_cols)
            self.scaler = fitted_scaler # Store fitted scaler
        elif not process_cols: self._log("No numeric features to scale (after exclusions).", level="info")
        else: # scale_numeric is False
            self._log("Scaling of numeric features skipped as per 'scale_numeric=False' configuration.", level="info")
            # Ensure metadata is set for unscaled processed columns
            for col in process_cols:
                if col in self.feature_type_metadata and '_scaled' not in self.feature_type_metadata[col]: pass # Already typed by transform
                elif col not in self.feature_type_metadata: self.feature_type_metadata[col] = 'numeric_processed_unscaled'


        self.fitted = True
        self.logger.info(f"--- Preprocessing Pipeline (fit_transform) Completed. Final shape: {_df.shape} ---")

        # Sanity checks post-processing
        if self.log_price_col in _df.columns:
            num_log_price_nan = _df[self.log_price_col].isnull().sum()
            if num_log_price_nan == len(_df) and n_positive > 0 : # Only warn if there *were* positive prices initially
                self.logger.warning(f"CRITICAL WARNING: Column '{self.log_price_col}' is all NaNs post-preprocessing, "
                                    f"even though positive prices were present in '{self.price_col}'!")
            elif num_log_price_nan > 0 :
                self._log(f"Note: Log-price column '{self.log_price_col}' contains {num_log_price_nan} NaNs "
                          f"post-preprocessing (expected for non-positive/missing original prices in '{self.price_col}').", level="info")
        else:
             self.logger.error(f"CRITICAL: Log-price column '{self.log_price_col}' is MISSING from DataFrame after fit_transform!")


        final_check_cols = [c for c in process_cols if c in _df.columns] # Re-check for existence
        if final_check_cols and _df[final_check_cols].isnull().any().any():
            nan_counts_final = _df[final_check_cols].isnull().sum()
            cols_with_nans_final = nan_counts_final[nan_counts_final > 0].index.tolist()
            self.logger.critical(f"CRITICAL: NaNs found in final processed numeric modeling columns: {cols_with_nans_final}. "
                                 f"Counts: {nan_counts_final[nan_counts_final > 0].to_dict()}. This may indicate issues in imputation within transform/scale.")
        return _df

    def transform(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted using 'fit_transform' before calling 'transform'.")

        self.logger.info(f"--- Applying Fitted Preprocessing Pipeline (transform) to new data. Shape: {df.shape} ---")
        _df = df.copy() # Operate on a copy

        # Price cleaning uses fitted cutoffs (if any) and fitted IF model. It also creates/updates log_price_col.
        _df = self.clean_and_transform_price(_df)
        if _df is None or _df.empty:
            self.logger.warning("DataFrame empty after price cleaning in transform. Returning as is."); return _df

        # BBL creation if it was done during fit
        if self.feature_type_metadata.get('bbl') == 'identifier_created':
            _df = self.create_bbl_identifier(_df)

        processed_in_transform = set() # Keep track of columns processed by specific transformers
        # Apply fitted feature transformers
        for transformer_key, transformer_instance in self.feature_transformers.items():
            # Infer columns this transformer was fitted on from its key
            # Assuming key format like "quantile_transformer_for_colA_colB"
            try:
                # Robustly extract columns from the key
                key_parts = transformer_key.split('_for_')
                transform_type_from_key = key_parts[0].split('_transformer')[0] if key_parts else "unknown"
                cols_str = key_parts[1] if len(key_parts) > 1 else ''

                # Find actual columns this transformer was fitted on (from self.transformed_features_list and metadata)
                # This is tricky if keys are not perfectly parseable. A better way is to store (transformer, cols_list) in self.feature_transformers.
                # For now, let's try to find relevant columns for this specific instance:
                cols_to_apply_transform = []
                if cols_str: # If key has _for_colA_colB structure
                    potential_cols = cols_str.split('_')
                    cols_to_apply_transform = [c for c in potential_cols if c in _df.columns and c not in processed_in_transform]
                else: # Fallback: if key is just 'quantile_transformer' (less specific)
                    cols_to_apply_transform = [
                        col for col, meta in self.feature_type_metadata.items()
                        if f'transformed_{transform_type_from_key}' in meta and col in _df.columns and col not in processed_in_transform
                        and col in self.transformed_features_list # Ensure it was one of the transformed columns
                    ]
                cols_to_apply_transform = list(set(cols_to_apply_transform)) # Unique

            except Exception as e_key_parse:
                 self._log(f"Could not parse columns from transformer key '{transformer_key}': {e_key_parse}. Skipping this transformer.", level="warning")
                 continue

            if not cols_to_apply_transform:
                self._log(f"No columns found or remaining in input df for transformer key '{transformer_key}'. Skipping.", level="debug")
                continue

            self._log(f"Applying fitted transformer '{transformer_key}' to columns: {cols_to_apply_transform}", level="debug")
            X_new_subset = _df[cols_to_apply_transform].copy() # Operate on copy of relevant columns

            # Impute NaNs using median of the current subset (as during fit)
            if X_new_subset.isnull().any().any():
                self._log(f"NaNs detected in new data for transform '{transformer_key}' for columns {cols_to_apply_transform}. Imputing with median of subset.", level="warning")
                for c in cols_to_apply_transform:
                    if X_new_subset[c].isnull().any():
                        median_val = X_new_subset[c].median()
                        X_new_subset[c].fillna(median_val if not pd.isna(median_val) else 0.0, inplace=True)

            # Replace Infs after imputation
            if not np.all(np.isfinite(X_new_subset.values)):
                self._log(f"Non-finite values detected after imputation for '{transformer_key}' on {cols_to_apply_transform}. Replacing with 0.", level="warning")
                X_new_subset.replace([np.inf, -np.inf], 0, inplace=True)

            # Special handling for Box-Cox (requires positive values)
            if isinstance(transformer_instance, PowerTransformer) and transformer_instance.method == 'box-cox':
                if np.any(X_new_subset.values <= 0):
                    self._log(f"Clipping non-positive values to 1e-6 for Box-Cox transform on {cols_to_apply_transform} in transform().", level="warning")
                    X_new_subset = X_new_subset.clip(lower=1e-6)
            try:
                transformed_values_new = transformer_instance.transform(X_new_subset)
                _df[cols_to_apply_transform] = transformed_values_new
                processed_in_transform.update(cols_to_apply_transform)
            except Exception as e:
                self._log(f"Error applying fitted transform '{transformer_key}' to {cols_to_apply_transform}: {e}. Data for these columns might remain untransformed.", level="error", exc_info=True)

        # Handle direct log1p if it wasn't a sklearn transformer (based on metadata from fit_transform)
        log1p_cols_from_meta = [col for col, meta in self.feature_type_metadata.items()
                                if meta == 'numeric_transformed_log1p' and col in _df.columns and col not in processed_in_transform]
        if log1p_cols_from_meta:
            self._log(f"Applying direct log1p transform (as done in fit) to: {log1p_cols_from_meta}", level="debug")
            for col in log1p_cols_from_meta:
                col_data = _df[col].copy()
                if np.any(col_data.values < 0): self._log(f"log1p (transform) encountered negative values in {col}. NaNs will be cleaned.", level="warning")
                _df[col] = np.log1p(col_data)
                _df[col] = np.nan_to_num(_df[col], nan=0.0, posinf=0.0, neginf=0.0) # Clean after log1p
                processed_in_transform.update([col])


        # Apply fitted scaler to all columns that were scaled during fit_transform
        if self.scaler and self.scaled_features_list:
            cols_to_scale_new = [f for f in self.scaled_features_list if f in _df.columns] # Only scale if present
            if cols_to_scale_new:
                X_to_scale_new = _df[cols_to_scale_new].copy() # Operate on copy

                # Impute NaNs using fitted means from the scaler if available, else median of current subset
                if X_to_scale_new.isnull().any().any():
                    cols_with_nan_transform = X_to_scale_new.columns[X_to_scale_new.isnull().any()].tolist()
                    self._log(f"NaNs detected before scaling new data: {cols_with_nan_transform}. Imputing...", level="warning")
                    for i, col_name in enumerate(cols_to_scale_new):
                        if X_to_scale_new[col_name].isnull().any():
                            impute_val = 0.0 # Default imputation
                            try: # Try to find original index for this column as per fitting order
                                original_training_idx = self.scaled_features_list.index(col_name)
                                if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None and original_training_idx < len(self.scaler.mean_):
                                    impute_val = self.scaler.mean_[original_training_idx]
                                else: # Fallback to current median if mean_ not suitable or col not found
                                    impute_val = X_to_scale_new[col_name].median()
                            except (ValueError, IndexError):
                                impute_val = X_to_scale_new[col_name].median() # Fallback

                            X_to_scale_new[col_name].fillna(impute_val if not pd.isna(impute_val) else 0.0, inplace=True)

                # Replace Infs after imputation
                if not np.all(np.isfinite(X_to_scale_new.values)):
                    self._log("Non-finite values (Inf) detected before scaling (transform) after NaN imputation. Replacing with 0.", level="error")
                    X_to_scale_new.replace([np.inf, -np.inf], 0, inplace=True)
                try:
                    scaled_values_new = self.scaler.transform(X_to_scale_new)
                    if np.isnan(scaled_values_new).any(): # Handle NaNs from scaling (e.g. constant cols in new data)
                        self._log("NaNs generated by scaler.transform (likely constant cols in new data or during fit). Replacing with 0.", level="warning")
                        scaled_values_new = np.nan_to_num(scaled_values_new, nan=0.0)
                    _df[cols_to_scale_new] = scaled_values_new
                    self._log(f"Applied fitted scaler to {len(cols_to_scale_new)} features.", level="info")
                except Exception as e:
                    self._log(f"Error applying fitted scaler: {e}. Data for these columns might remain unscaled.", level="error", exc_info=True)
            else: self._log("No features designated for scaling found in the new data.", level="debug")
        else: self._log("No scaler fitted or no features were designated for scaling during fit. Skipping scaling in transform.", level="debug")

        self.logger.info(f"--- Preprocessing Pipeline (transform) Completed. Final shape: {_df.shape} ---")
        return _df

