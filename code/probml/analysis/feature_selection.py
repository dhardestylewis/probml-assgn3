# -*- coding: utf-8 -*-
from probml.core.utils import get_logger

import pandas as pd
import numpy as np
import warnings
import time
import logging
from typing import List, Optional, Dict, Tuple, Union, Any
from collections import defaultdict

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_selection import mutual_info_classif, f_classif, VarianceThreshold
from scipy.sparse import diags as sparse_diags
from scipy.sparse import csgraph, csr_matrix # csgraph is used for graph laplacian
from sklearn.utils import resample # For bootstrapping
from sklearn.cluster import KMeans

# --- Global FAISS availability check ---
FAISS_AVAILABLE = False
faiss = None # Define faiss as None initially
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass


class FeatureSelectorConfig:
    """Configuration class for FeatureSelector."""
    def __init__(self,
                 # General thresholds
                 variance_threshold: float = 1e-4,
                 correlation_threshold: Optional[float] = 0.9, # Set to None or >= 1.0 to disable
                 methods_to_run: List[str] = ['variance', 'laplacian', 'pseudo_label'],
                 # Coverage-based k selection (New/Updated as per spec)
                 coverage_threshold: float = 0.9, # Target cumulative importance
                 # Bootstrap stability selection (New/Updated as per spec)
                 bootstrap_B: int = 30, # Number of bootstrap iterations
                 bootstrap_k_per_iter: int = 15, # k for the .select() call within each bootstrap
                 bootstrap_selection_freq_threshold: Optional[float] = 0.7, # e.g., 0.7 (70% of bootstraps)
                 # Laplacian score parameters
                 laplacian_n_neighbors: int = 15,
                 laplacian_sample_size: Optional[int] = 50000,
                 laplacian_weight_mode: str = 'heat', # 'heat' or 'connectivity'
                 laplacian_gamma: float = 1.0, # For 'heat' mode (sigma for Gaussian kernel)
                 laplacian_use_faiss: bool = True, # Attempt to use FAISS if available
                 laplacian_prefer_gpu: bool = False, # If using FAISS, attempt to use GPU
                 # Pseudo-label ensemble parameters
                 pseudo_label_n_clusters_list: List[int] = [3, 5, 8],
                 pseudo_label_methods: List[str] = ['kmeans'], # e.g. ['kmeans', 'agglomerative'] from ClusteringSuite
                 pseudo_label_scoring: str = 'f_classif', # 'f_classif' or 'mi' (mutual information)
                 pseudo_label_sample_size: Optional[int] = 50000,
                 use_clustering_suite_for_pseudo: bool = True # Attempt to use ClusteringSuite if available
                 ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.methods_to_run = methods_to_run
        self.coverage_threshold = coverage_threshold
        self.bootstrap_B = bootstrap_B
        self.bootstrap_k_per_iter = bootstrap_k_per_iter
        self.bootstrap_selection_freq_threshold = bootstrap_selection_freq_threshold
        self.laplacian_n_neighbors = laplacian_n_neighbors
        self.laplacian_sample_size = laplacian_sample_size
        self.laplacian_weight_mode = laplacian_weight_mode
        self.laplacian_gamma = laplacian_gamma
        self.laplacian_use_faiss = laplacian_use_faiss
        self.laplacian_prefer_gpu = laplacian_prefer_gpu
        self.pseudo_label_n_clusters_list = pseudo_label_n_clusters_list
        self.pseudo_label_methods = pseudo_label_methods
        self.pseudo_label_scoring = pseudo_label_scoring
        self.pseudo_label_sample_size = pseudo_label_sample_size
        self.use_clustering_suite_for_pseudo = use_clustering_suite_for_pseudo

        # Basic validation
        if not (0 <= self.coverage_threshold <= 1):
            raise ValueError("coverage_threshold must be between 0 and 1.")
        if self.correlation_threshold is not None and \
           not (0 <= self.correlation_threshold <= 1) and \
           self.correlation_threshold < 0: # Note: if threshold is >1 it's disabled, so only check <0 here
            raise ValueError("correlation_threshold must be non-negative or None.")
        if self.bootstrap_selection_freq_threshold is not None and \
           not (0 <= self.bootstrap_selection_freq_threshold <= 1):
            raise ValueError("bootstrap_selection_freq_threshold must be between 0 and 1.")
        if self.bootstrap_k_per_iter <= 0:
            raise ValueError("bootstrap_k_per_iter must be positive.")
        if self.bootstrap_B <= 0:
            raise ValueError("bootstrap_B must be positive.")


class FeatureSelector:
    """
    Selects relevant numeric features using a combination of methods.
    Supports fixed-k, coverage-based, or stability-based selection.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 id_cols: Optional[List[str]] = None,
                 target_col: Optional[str] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 config: Optional[FeatureSelectorConfig] = None):

        if df.empty:
            raise ValueError("Input DataFrame `df` cannot be empty.")

        self.df = df.copy() # Internal working copy
        self.original_df_for_bootstrap = df # Keep original for bootstrap, DO NOT MODIFY
        self.id_cols = id_cols if id_cols is not None else []
        self.target_col = target_col
        self.random_state = random_state
        self.verbose = verbose # Verbosity of this specific FeatureSelector instance
        self.n_jobs = n_jobs
        self.logger = get_logger(self.__class__.__name__, verbose)
        self.config = config if config is not None else FeatureSelectorConfig()

        self.feature_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.datetime_cols: List[str] = []

        self._identify_column_types()
        self._initial_clean() # Clean self.df

        self.logger.info(
            f"FeatureSelector initialized. Numeric: {len(self.numeric_cols)}, "
            f"Categorical: {len(self.categorical_cols)}, Datetime: {len(self.datetime_cols)}."
        )
        self.logger.debug(f"Config: {vars(self.config)}")

    def _identify_column_types(self):
        """Identifies numeric, categorical, and datetime columns, excluding ID/target."""
        self.logger.info("Identifying column types...")
        potential_feature_cols = [
            col for col in self.df.columns
            if col not in self.id_cols and col != self.target_col
        ]
        potential_feature_cols = [col for col in potential_feature_cols if col in self.df.columns]

        self.numeric_cols = self.df[potential_feature_cols].select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df[potential_feature_cols].select_dtypes(include=['category', 'object', 'string']).columns.tolist()
        self.datetime_cols = self.df[potential_feature_cols].select_dtypes(include=['datetime', 'datetimetz', 'timedelta']).columns.tolist()

        self.logger.info(
            f"Identified - Numeric: {len(self.numeric_cols)}, "
            f"Categorical: {len(self.categorical_cols)}, Datetime: {len(self.datetime_cols)}"
        )
        self.logger.debug(f"Numeric cols found: {self.numeric_cols}")

    def _initial_clean(self):
        """Initial NaN/Inf imputation on the internal DataFrame copy for numeric columns."""
        self.logger.info("Performing initial NaN/Inf cleaning for numeric columns on internal df copy...")
        num_nan_imputed = 0
        num_inf_imputed = 0

        if self.numeric_cols:
            for col in self.numeric_cols:
                col_data = self.df[col]
                if col_data.isnull().any():
                    median_val = col_data.median()
                    fill_nan_val = median_val if pd.notna(median_val) else 0.0
                    self.df[col].fillna(fill_nan_val, inplace=True)
                    num_nan_imputed += 1

                if np.isinf(self.df[col].values).any():
                    finite_vals = self.df[col][np.isfinite(self.df[col])]
                    fill_inf_val = finite_vals.median() if not finite_vals.empty else 0.0
                    self.df[col].replace([np.inf, -np.inf], fill_inf_val, inplace=True)
                    num_inf_imputed += 1

            if num_nan_imputed > 0:
                self.logger.info(f"Imputed NaNs in {num_nan_imputed} numeric columns using median.")
            if num_inf_imputed > 0:
                self.logger.info(f"Replaced Infs in {num_inf_imputed} numeric columns using median of finite values.")
        else:
            self.logger.info("No numeric columns to clean initially.")

    def filter_by_variance(self) -> List[str]:
        """Filters features based on variance threshold."""
        threshold = self.config.variance_threshold
        self.logger.info(f"Filtering features with variance < {threshold}...")

        if not self.numeric_cols:
            self.logger.warning("No numeric columns for variance filtering.")
            return []

        df_numeric = self.df[self.numeric_cols].copy()
        cleaned_cols_count = 0
        for col in df_numeric.columns:
            if not np.all(np.isfinite(df_numeric[col])):
                finite_vals = df_numeric[col][np.isfinite(df_numeric[col])]
                fill_val = finite_vals.median() if not finite_vals.empty else 0.0
                df_numeric[col] = np.nan_to_num(df_numeric[col], nan=fill_val, posinf=fill_val, neginf=fill_val)
                cleaned_cols_count += 1
        if cleaned_cols_count > 0:
            self.logger.debug(f"Ensured finite values in {cleaned_cols_count} columns before variance check.")

        selector = VarianceThreshold(threshold=threshold)
        try:
            selector.fit(df_numeric)
            selected_mask = selector.get_support()
            selected_cols = df_numeric.columns[selected_mask].tolist()
            removed_cols = df_numeric.columns[~selected_mask].tolist()

            removed_log_msg = f"Removed {len(removed_cols)} low-variance columns"
            if removed_cols:
                removed_log_msg += f": {removed_cols[:10]}{'...' if len(removed_cols) > 10 else ''}"
            self.logger.info(removed_log_msg)
            self.logger.info(f"{len(selected_cols)} numeric columns remain after variance filtering.")

            # Update internal state AND store scores for selected features
            # self.numeric_cols = selected_cols # This should not update self.numeric_cols but return the list
                                             # The caller (.select) will manage current_numeric_features
            valid_variances = selector.variances_[selected_mask]
            for i, col_name in enumerate(selected_cols):
                self.feature_scores[col_name]['variance'] = valid_variances[i]
            return selected_cols

        except Exception as e:
            self.logger.error(f"Error during variance filtering: {e}. Returning pre-filter numeric columns.", exc_info=True)
            # Fallback to numeric columns that were present before this filter attempt
            return [col for col in self.df.columns if col in self.numeric_cols]


    def _build_knn_graph(self,
                         X_scaled: np.ndarray,
                         n_neighbors: int,
                         use_faiss_cfg: bool,
                         prefer_gpu_cfg: bool) -> Optional[csr_matrix]:
        """Builds k-NN graph using FAISS or sklearn."""
        n_samples = X_scaled.shape[0]

        if n_samples <= n_neighbors:
            n_neighbors = max(1, n_samples - 1)
            self.logger.warning(f"Reduced k-NN k to {n_neighbors} because n_samples ({n_samples}) <= k.")
        if n_neighbors == 0 and n_samples > 0:
            n_neighbors = 1
        if n_neighbors <= 0 or n_samples == 0:
            self.logger.error("Cannot build k-NN graph with n_samples=0 or k<=0.")
            return None

        t_start = time.time()

        if use_faiss_cfg and FAISS_AVAILABLE and faiss and X_scaled.ndim == 2 and X_scaled.shape[1] > 0:
            self.logger.info(f"Building k-NN graph using FAISS (k={n_neighbors}).")
            index_cpu = None
            index_gpu = None
            gpu_resources = None
            index_to_use = None
            try:
                d = X_scaled.shape[1]
                X_scaled_32 = X_scaled.astype('float32')
                index_cpu = faiss.IndexFlatL2(d)
                index_to_use = index_cpu

                if prefer_gpu_cfg and hasattr(faiss, 'StandardGpuResources'):
                    try:
                        gpu_resources = faiss.StandardGpuResources()
                        index_gpu = faiss.index_cpu_to_gpu(gpu_resources, 0, index_cpu)
                        index_to_use = index_gpu
                        self.logger.info("FAISS is using GPU.")
                    except Exception as gpu_e:
                        self.logger.warning(f"FAISS GPU initialization failed: {gpu_e}. Falling back to CPU.")
                        index_to_use = index_cpu
                elif prefer_gpu_cfg:
                    self.logger.info("FAISS GPU preferred but StandardGpuResources not available. Using CPU.")

                index_to_use.add(X_scaled_32)
                distances, indices = index_to_use.search(X_scaled_32, n_neighbors + 1)

                rows = np.arange(n_samples).repeat(n_neighbors)
                cols = indices[:, 1:].flatten()
                data = distances[:, 1:].flatten()

                valid_mask = (cols >= 0) & (cols < n_samples)
                rows, cols, data = rows[valid_mask], cols[valid_mask], data[valid_mask]

                if data.dtype.kind == 'f':
                    data[~np.isfinite(data)] = np.finfo(data.dtype).max
                else:
                    data[~np.isfinite(data)] = 0

                graph = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
                self.logger.info(f"FAISS k-NN graph built in {time.time() - t_start:.2f}s.")
                return graph
            except Exception as faiss_e:
                self.logger.warning(f"FAISS k-NN graph construction failed: {faiss_e}. Falling back to sklearn.", exc_info=True)
            finally:
                del index_cpu, index_gpu, gpu_resources, index_to_use

        self.logger.info(f"Building k-NN graph using sklearn (k={n_neighbors}).")
        try:
            graph = kneighbors_graph(X_scaled,
                                     n_neighbors=n_neighbors,
                                     mode='distance',
                                     include_self=False,
                                     n_jobs=self.n_jobs,
                                     algorithm='auto')
            self.logger.info(f"Sklearn k-NN graph built in {time.time() - t_start:.2f}s.")
            return graph
        except Exception as sk_e:
            self.logger.error(f"Sklearn kneighbors_graph failed: {sk_e}", exc_info=True)
            return None

    def calculate_laplacian_score(self, features: List[str]) -> None:
        """Calculates Laplacian score for unsupervised feature selection."""
        cfg = self.config
        self.logger.info(
            f"Calculating Laplacian Scores (k={cfg.laplacian_n_neighbors}, "
            f"sample_size={cfg.laplacian_sample_size}, weight_mode='{cfg.laplacian_weight_mode}')..."
        )

        if not features:
            self.logger.warning("No features provided for Laplacian Score calculation.")
            return

        df_subset = self.df # Use the cleaned self.df
        if cfg.laplacian_sample_size is not None and \
           0 < cfg.laplacian_sample_size < len(df_subset):
            actual_sample_size = min(cfg.laplacian_sample_size, len(df_subset))
            self.logger.info(f"Subsampling to {actual_sample_size} rows for Laplacian Score.")
            if actual_sample_size <= cfg.laplacian_n_neighbors:
                self.logger.warning(
                    f"Sample size ({actual_sample_size}) is less than or equal to "
                    f"k_neighbors ({cfg.laplacian_n_neighbors}). This might lead to issues."
                )
            sampler_rng = np.random.default_rng(self.random_state)
            indices_used = sampler_rng.choice(df_subset.index, size=actual_sample_size, replace=False)
            df_subset = df_subset.loc[indices_used]

        X = df_subset[features].values

        if X.shape[0] <= 1 or X.shape[1] == 0:
            self.logger.warning(f"Not enough data (Shape: {X.shape}) for Laplacian score calculation. Skipping.")
            for col_name in features: self.feature_scores[col_name]['laplacian_score'] = np.inf
            return

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            distance_graph = self._build_knn_graph(
                X_scaled,
                cfg.laplacian_n_neighbors,
                cfg.laplacian_use_faiss,
                cfg.laplacian_prefer_gpu
            )
            if distance_graph is None:
                self.logger.error("Failed to build k-NN graph for Laplacian Score. Scores will be set to inf.")
                for col_name in features: self.feature_scores[col_name]['laplacian_score'] = np.inf
                return

            if cfg.laplacian_weight_mode == 'heat':
                W = distance_graph.copy()
                W.data = np.exp(- (W.data**2) / (2 * cfg.laplacian_gamma**2))
            else: # 'connectivity' mode
                W = distance_graph.copy()
                W.data[:] = 1.0

            W = 0.5 * (W + W.T)

            D_values = np.array(W.sum(axis=1)).flatten()
            if np.all(np.abs(D_values) < 1e-9):
                self.logger.error("Graph appears to be disconnected (all D_values are near zero). Laplacian scores will be inf.")
                for col_name in features: self.feature_scores[col_name]['laplacian_score'] = np.inf
                return
            D_sparse = sparse_diags(D_values)

            L_graph = D_sparse - W

            for i, col_name in enumerate(features):
                f_r = X[:, i].astype(float)
                if np.isnan(f_r).any(): # Should be handled by _initial_clean, but safeguard
                    median_val_col = self.df[col_name].median()
                    f_r = np.nan_to_num(f_r, nan=median_val_col if pd.notna(median_val_col) else 0.0)

                sum_D = D_values.sum()
                if abs(sum_D) < 1e-9:
                    f_r_mean = np.mean(f_r)
                else:
                    f_r_mean = np.dot(f_r, D_values) / sum_D

                f_r_tilde = f_r - f_r_mean

                numerator = f_r_tilde.T @ L_graph @ f_r_tilde
                denominator = f_r_tilde.T @ D_sparse @ f_r_tilde

                if np.abs(denominator) < 1e-9:
                    score = np.inf
                else:
                    score = numerator / denominator
                self.feature_scores[col_name]['laplacian_score'] = score

            self.logger.info("Laplacian Score calculation complete.")

        except Exception as e:
            self.logger.error(f"Error during Laplacian Score calculation: {e}", exc_info=True)
            for col_name in features:
                self.feature_scores[col_name]['laplacian_score'] = np.inf


    def calculate_ensemble_pseudo_label_scores(self, features: List[str]) -> None:
        """Calculates feature scores based on pseudo-labels from multiple clustering runs."""
        cfg = self.config
        self.logger.info(
            f"Calculating Ensemble Pseudo-Label Scores (K-values: {cfg.pseudo_label_n_clusters_list}, "
            f"Methods: {cfg.pseudo_label_methods}, Scoring: '{cfg.pseudo_label_scoring}', "
            f"Sample: {cfg.pseudo_label_sample_size})..."
        )

        if not features:
            self.logger.warning("No features provided for pseudo-label score calculation.")
            return

        df_subset = self.df # Use the cleaned self.df
        if cfg.pseudo_label_sample_size is not None and \
           0 < cfg.pseudo_label_sample_size < len(df_subset):
            actual_sample_size = min(cfg.pseudo_label_sample_size, len(df_subset))
            self.logger.info(f"Subsampling to {actual_sample_size} rows for pseudo-label generation.")

            max_k_val = max(cfg.pseudo_label_n_clusters_list if cfg.pseudo_label_n_clusters_list else [2])
            if actual_sample_size <= max_k_val:
                self.logger.warning(
                    f"Sample size ({actual_sample_size}) is less than or equal to max K ({max_k_val}). "
                    "This may not be ideal for clustering. Consider increasing sample_size or decreasing K."
                )
            sampler_rng = np.random.default_rng(self.random_state)
            indices_used = sampler_rng.choice(df_subset.index, size=actual_sample_size, replace=False)
            df_subset = df_subset.loc[indices_used]

        if df_subset.empty:
            self.logger.error("Subsampled DataFrame for pseudo-labels is empty. Skipping.")
            return
        X_for_scoring = df_subset[features].values
        if X_for_scoring.shape[0] == 0:
            self.logger.error("No data rows available for pseudo-label scoring after selecting features. Skipping.")
            return

        max_k_val_check = max(cfg.pseudo_label_n_clusters_list if cfg.pseudo_label_n_clusters_list else [2])
        if X_for_scoring.shape[0] <= max_k_val_check:
            self.logger.warning(
                f"Number of samples ({X_for_scoring.shape[0]}) is less than or equal to the "
                f"maximum k for clustering ({max_k_val_check}). Skipping pseudo-label scoring."
            )
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_for_scoring)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        feature_scores_aggregate = defaultdict(list)

        _use_suite = cfg.use_clustering_suite_for_pseudo and CLUSTERING_SUITE_AVAILABLE and ClusteringSuite is not None
        clusterer_instance = None
        if _use_suite:
            try:
                clusterer_instance = ClusteringSuite(
                    random_state=self.random_state,
                    verbose=False,
                    n_jobs=self.n_jobs
                )
                self.logger.info("Using ClusteringSuite for pseudo-label generation.")
            except Exception as e_cs:
                self.logger.error(f"Failed to initialize ClusteringSuite: {e_cs}. Falling back to basic KMeans.", exc_info=False)
                _use_suite = False

        for k_clusters in cfg.pseudo_label_n_clusters_list:
            if k_clusters <= 1 or k_clusters >= X_scaled.shape[0]:
                self.logger.warning(f"Skipping k={k_clusters} for pseudo-labels (invalid for {X_scaled.shape[0]} samples).")
                continue

            for method_name in cfg.pseudo_label_methods:
                labels = None
                self.logger.debug(f"Generating pseudo-labels: Method='{method_name}', k={k_clusters}")
                try:
                    if _use_suite and clusterer_instance:
                        if method_name == 'kmeans' and hasattr(clusterer_instance, 'run_kmeans'):
                            labels = clusterer_instance.run_kmeans(X_scaled, k_clusters)
                        elif method_name == 'agglomerative' and hasattr(clusterer_instance, 'run_agglomerative'):
                            labels = clusterer_instance.run_agglomerative(X_scaled, k_clusters)
                        else:
                            self.logger.warning(f"Method '{method_name}' not available in configured ClusteringSuite. Skipping for k={k_clusters}.")
                            continue
                    elif method_name == 'kmeans':
                        kmeans = KMeans(n_clusters=k_clusters, random_state=self.random_state, n_init='auto')
                        labels = kmeans.fit_predict(X_scaled)
                    else:
                        self.logger.warning(f"Unsupported pseudo-label method '{method_name}' without ClusteringSuite. Skipping for k={k_clusters}.")
                        continue
                except Exception as e_clust:
                    self.logger.warning(f"Clustering method '{method_name}' with k={k_clusters} failed: {e_clust}", exc_info=False)
                    continue

                if labels is not None and len(np.unique(labels)) > 1:
                    try:
                        current_run_scores = []
                        if cfg.pseudo_label_scoring == 'f_classif':
                            f_values, _ = f_classif(X_scaled, labels)
                            current_run_scores = np.nan_to_num(f_values, nan=0.0)
                        elif cfg.pseudo_label_scoring == 'mi':
                            n_bins = max(3, min(10, int(np.sqrt(X_scaled.shape[0]) / 5)))
                            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
                            try:
                                X_discretized = discretizer.fit_transform(X_scaled)
                                current_run_scores = mutual_info_classif(X_discretized, labels, discrete_features=True, random_state=self.random_state)
                            except ValueError as ve:
                                self.logger.warning(f"Discretization or MI calculation failed for k={k_clusters} (scoring: {cfg.pseudo_label_scoring}): {ve}. Setting scores to 0 for this run.")
                                current_run_scores = np.zeros(X_scaled.shape[1])
                        else:
                            self.logger.warning(f"Unsupported pseudo_label_scoring '{cfg.pseudo_label_scoring}'. Skipping scoring for this run.")
                            continue

                        for i, col_name in enumerate(features):
                            feature_scores_aggregate[col_name].append(current_run_scores[i])
                    except Exception as e_score:
                        self.logger.warning(f"Score calculation (type: {cfg.pseudo_label_scoring}) failed for k={k_clusters}, method={method_name}: {e_score}", exc_info=False)
                elif labels is not None:
                    self.logger.warning(f"Clustering with method '{method_name}', k={k_clusters} resulted in < 2 unique labels. Skipping scoring for this run.")

        score_key = f'ensemble_pseudo_label_{cfg.pseudo_label_scoring}'
        for col_name in features:
            scores_list = feature_scores_aggregate[col_name]
            if scores_list:
                avg_score = np.nanmean(scores_list)
                self.feature_scores[col_name][score_key] = avg_score if np.isfinite(avg_score) else 0.0
            else:
                self.feature_scores[col_name][score_key] = 0.0
                self.logger.debug(f"No pseudo-label scores collected for feature '{col_name}'. Setting to 0.")

        self.logger.info(f"Ensemble Pseudo-Label Score (using '{cfg.pseudo_label_scoring}') calculation complete.")

    def aggregate_scores(self) -> pd.DataFrame:
        """Aggregates scores from different methods into a single importance score and rank."""
        self.logger.info("Aggregating feature scores...")

        if not self.feature_scores:
            self.logger.warning("No feature scores available to aggregate.")
            return pd.DataFrame()

        all_features_with_scores = list(self.feature_scores.keys())
        if not all_features_with_scores:
             self.logger.warning("Feature scores dictionary is populated but has no feature keys.")
             return pd.DataFrame()

        try:
            score_df = pd.DataFrame.from_dict(self.feature_scores, orient='index')
        except Exception as e:
            self.logger.error(f"Failed to create DataFrame from feature_scores: {e}. Feature scores: {self.feature_scores}")
            return pd.DataFrame()

        if score_df.empty:
            self.logger.warning("Created score_df is empty. No scores to aggregate.")
            return pd.DataFrame()

        ascending_score_types = ['laplacian_score']
        ranks_df = pd.DataFrame(index=score_df.index)
        ranked_any_method = False

        for score_name in score_df.columns:
            if score_df[score_name].notna().any() and \
               np.isfinite(score_df[score_name].replace([np.inf, -np.inf], np.nan)).any():

                series_to_rank = score_df[score_name].copy()
                is_ascending = score_name in ascending_score_types

                if is_ascending:
                    series_to_rank.replace(np.inf, np.nan, inplace=True)
                    # Smallest finite value or a large negative if all are inf/nan
                    fill_neg_inf_val = (series_to_rank[np.isfinite(series_to_rank)].min() - 1
                                        if np.isfinite(series_to_rank).any() else -1e12)
                    series_to_rank.replace(-np.inf, fill_neg_inf_val, inplace=True)
                else: # Descending rank
                    series_to_rank.replace(-np.inf, np.nan, inplace=True)
                    # Largest finite value or a large positive if all are inf/nan
                    fill_pos_inf_val = (series_to_rank[np.isfinite(series_to_rank)].max() + 1
                                         if np.isfinite(series_to_rank).any() else 1e12)
                    series_to_rank.replace(np.inf, fill_pos_inf_val, inplace=True)

                ranks_df[f'{score_name}_rank'] = series_to_rank.rank(
                    ascending=is_ascending,
                    na_option='bottom',
                    method='average'
                )
                ranked_any_method = True
            else:
                self.logger.debug(f"Score type '{score_name}' has no valid finite values to rank. It will not contribute to aggregated rank.")
                ranks_df[f'{score_name}_rank'] = np.nan

        if not ranked_any_method:
            self.logger.warning("No scores could be ranked from any method. Returning empty aggregated DataFrame.")
            return pd.DataFrame()

        max_rank_across_methods = ranks_df.max().max() if not ranks_df.isnull().all().all() else len(ranks_df)
        ranks_df.fillna(max_rank_across_methods + 1, inplace=True)

        score_df['aggregated_rank'] = ranks_df.mean(axis=1)

        if score_df['aggregated_rank'].isnull().any():
             self.logger.warning("Aggregated rank contains NaNs, which is unexpected. Filling with max possible rank + 1.")
             fill_val = (score_df['aggregated_rank'].max() + 1
                         if score_df['aggregated_rank'].notna().any() else len(score_df) + 1)
             score_df['aggregated_rank'].fillna(fill_val, inplace=True)

        max_aggregated_rank = score_df['aggregated_rank'].max()
        score_df['importance_score'] = (max_aggregated_rank - score_df['aggregated_rank']) + 1

        score_df.sort_values('aggregated_rank', ascending=True, inplace=True)
        score_df.reset_index(inplace=True)
        score_df.rename(columns={'index': 'feature'}, inplace=True)

        self.logger.info("Score aggregation complete.")
        if not score_df.empty:
             self.logger.info(f"Top 5 features by aggregated rank:\n{score_df[['feature', 'aggregated_rank', 'importance_score']].head()}")
        else:
             self.logger.warning("Aggregated score DataFrame is empty after processing.")

        cols_to_return = ['feature', 'aggregated_rank', 'importance_score']
        if all_features_with_scores:
            first_feature_key = all_features_with_scores[0]
            original_score_keys = list(self.feature_scores.get(first_feature_key, {}).keys())
            cols_to_return.extend([col for col in original_score_keys if col in score_df.columns and col not in cols_to_return])
        cols_to_return.extend([col for col in ranks_df.columns if col in score_df.columns and col not in cols_to_return])

        final_cols = []
        for col in cols_to_return:
            if col not in final_cols:
                final_cols.append(col)
        for col in score_df.columns:
            if col not in final_cols:
                 final_cols.append(col)

        return score_df[final_cols]

    def _prune_highly_correlated_features(self,
                                          features_df: pd.DataFrame,
                                          data_for_correlation: pd.DataFrame) -> pd.DataFrame:
        """
        Prunes highly correlated features. Keeps the one with better (lower) aggregated_rank.
        If ranks are tied, keeps the one with lower mean absolute correlation with other features.
        `features_df` must contain 'feature' and 'aggregated_rank'.
        `data_for_correlation` is the DataFrame to compute correlations from (e.g., self.df or a bootstrap sample).
        """
        threshold = self.config.correlation_threshold
        if threshold is None or threshold >= 1.0:
            self.logger.info("Correlation pruning skipped as threshold is None or >= 1.0.")
            return features_df.copy()

        if features_df.empty or 'feature' not in features_df.columns or 'aggregated_rank' not in features_df.columns:
            self.logger.warning("Cannot prune correlated features: input features_df is empty or missing required columns ('feature', 'aggregated_rank').")
            return features_df.copy() # Return copy to avoid side effects

        self.logger.info(f"Pruning features with absolute correlation > {threshold}...")

        ranked_features = features_df['feature'].tolist()
        features_in_data = [f for f in ranked_features if f in data_for_correlation.columns]

        if len(features_in_data) < 2:
            self.logger.info("Not enough features present in data_for_correlation to compute correlation matrix. Skipping pruning.")
            return features_df.copy()

        df_clean_for_corr = data_for_correlation[features_in_data].copy()
        cols_to_exclude_from_corr = set()

        for col in features_in_data: # Use features_in_data here
            if not np.all(np.isfinite(df_clean_for_corr[col])):
                finite_vals = df_clean_for_corr[col][np.isfinite(df_clean_for_corr[col])]
                fill_val = finite_vals.median() if not finite_vals.empty else 0.0
                df_clean_for_corr[col] = np.nan_to_num(df_clean_for_corr[col], nan=fill_val, posinf=fill_val, neginf=fill_val)
                if not np.all(np.isfinite(df_clean_for_corr[col])):
                    cols_to_exclude_from_corr.add(col)
                    self.logger.warning(f"Excluding column '{col}' from correlation calculation as it could not be made finite.")

        final_features_for_corr = [f for f in features_in_data if f not in cols_to_exclude_from_corr]

        if len(final_features_for_corr) < 2:
            self.logger.info("Not enough features remain after cleaning for correlation matrix computation. Skipping pruning.")
            return features_df.copy()

        corrmat = df_clean_for_corr[final_features_for_corr].corr().abs()
        upper_tri = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))
        features_to_drop = set()

        # Iterate based on features_df's rank order to ensure consistent choices when ranks are different
        # The provided features_df is already sorted by 'aggregated_rank'
        sorted_features_for_pruning = features_df[features_df['feature'].isin(final_features_for_corr)]['feature'].tolist()

        for i in range(len(sorted_features_for_pruning)):
            f1_name = sorted_features_for_pruning[i]
            if f1_name in features_to_drop: # If already decided to drop f1, skip
                continue
            for j in range(i + 1, len(sorted_features_for_pruning)):
                f2_name = sorted_features_for_pruning[j]
                if f2_name in features_to_drop: # If already decided to drop f2, skip
                    continue

                # Check if f1_name and f2_name are in upper_tri (which they should be if in final_features_for_corr)
                if f1_name not in upper_tri.index or f2_name not in upper_tri.columns:
                    # This might happen if one of them was excluded due to all NaNs, etc.
                    # Or if matrix construction had an issue.
                    self.logger.debug(f"Skipping correlation check between {f1_name} and {f2_name} due to absence in correlation matrix parts.")
                    continue


                # Determine which way to access from upper_tri (it's symmetric in value after .abs())
                # Ensure f1_name is row index and f2_name is column index, and f1_name < f2_name (lexicographically for consistency if not triangular)
                # However, upper_tri is already set up to have values only in the upper triangle.
                # We need to ensure we query it correctly (row_idx < col_idx generally)
                # The easiest is to get the value regardless of which is row/col from the full corrmat
                correlation_value = corrmat.loc[f1_name, f2_name]

                if pd.isna(correlation_value) or correlation_value <= threshold:
                    continue

                # Both features are highly correlated, decide which one to drop
                try:
                    # features_df is indexed by numbers, use .loc with boolean condition
                    rank_f1 = features_df.loc[features_df['feature'] == f1_name, 'aggregated_rank'].iloc[0]
                    rank_f2 = features_df.loc[features_df['feature'] == f2_name, 'aggregated_rank'].iloc[0]
                except IndexError:
                    self.logger.warning(f"Could not find rank for '{f1_name}' or '{f2_name}' in features_df. Skipping this pair for pruning.")
                    continue

                feature_to_drop_this_pair = None
                tie_break_info = ""

                # Lower rank is better. Drop the one with the WORSE (higher) rank.
                if rank_f1 > rank_f2:
                    feature_to_drop_this_pair = f1_name
                elif rank_f2 > rank_f1:
                    feature_to_drop_this_pair = f2_name
                else: # Ranks are tied, use mean absolute correlation as tie-breaker
                    # Ensure columns used for mean calculation are present in corrmat
                    valid_cols_for_f1 = [col for col in corrmat.columns if col != f1_name and col in corrmat.index] # Check col in corrmat.index too
                    valid_cols_for_f2 = [col for col in corrmat.columns if col != f2_name and col in corrmat.index]

                    mean_corr_f1 = corrmat.loc[f1_name, valid_cols_for_f1].mean() if valid_cols_for_f1 else np.inf
                    mean_corr_f2 = corrmat.loc[f2_name, valid_cols_for_f2].mean() if valid_cols_for_f2 else np.inf

                    tie_break_info = (f"Ranks tied ({rank_f1:.2f}). Mean Abs Corr: "
                                      f"{f1_name}({mean_corr_f1:.3f}), {f2_name}({mean_corr_f2:.3f}).")

                    # Drop the one with HIGHER mean absolute correlation (more redundant)
                    if mean_corr_f1 >= mean_corr_f2: # If tied here, f1 is dropped (consistent due to outer loop order)
                        feature_to_drop_this_pair = f1_name
                    else:
                        feature_to_drop_this_pair = f2_name

                if feature_to_drop_this_pair:
                    features_to_drop.add(feature_to_drop_this_pair)
                    self.logger.debug(
                        f"Correlation > {threshold}: {f1_name} vs {f2_name} ({correlation_value:.2f}). "
                        f"{tie_break_info} Dropping '{feature_to_drop_this_pair}'."
                    )
                    # If f1_name was dropped, break inner loop and continue outer,
                    # because f1_name (the one with better rank or chosen by tie-break if ranks equal) is kept.
                    # No, if feature_to_drop_this_pair is f1_name, we add f1_name to features_to_drop.
                    # The outer loop already handles `if f1_name in features_to_drop: continue`

        if features_to_drop:
            self.logger.info(f"Removing {len(features_to_drop)} features due to high correlation: {sorted(list(features_to_drop))}")
            return features_df[~features_df['feature'].isin(features_to_drop)].copy()
        else:
            self.logger.info("No features pruned by correlation.")
            return features_df.copy()


    def select(self,
               k: Optional[int] = None,
               methods_to_run_override: Optional[List[str]] = None,
               config_overrides: Optional[Dict[str, Any]] = None,
               auto_k_selection_method: Optional[str] = None) -> List[str]:
        """
        Main method to run the feature selection pipeline.
        Selects features based on fixed-k, coverage, or stability.
        """
        original_config_obj = self.config
        temp_config_dict = vars(original_config_obj).copy()

        if config_overrides:
            temp_config_dict.update(config_overrides)
        if methods_to_run_override is not None:
            temp_config_dict['methods_to_run'] = methods_to_run_override

        # Temporarily use the potentially modified config for this run
        # Create a new config object to avoid modifying the instance's config if this is called multiple times
        # or from within a bootstrap loop that shouldn't affect the parent's config.
        current_run_config = FeatureSelectorConfig(**temp_config_dict)

        # Store the current config and replace it with the temporary one for the duration of this method
        # This is tricky if self.config is used by other methods called herein, they need to see the temp one.
        # A better way is to pass current_run_config to sub-methods if they need it, or just use current_run_config locally.
        # For simplicity, and as self.config is used by helper methods, we'll swap it.
        _original_instance_config = self.config
        self.config = current_run_config

        self.logger.info(f"Running select() with effective config: {vars(self.config)}")

        selected_features: List[str] = []

        try:
            if auto_k_selection_method:
                method_lower = auto_k_selection_method.lower()
                self.logger.info(f"--- Automatic Feature Selection by: {method_lower} ---")
                if method_lower == "coverage":
                    selected_features = self.select_k_by_coverage()
                elif method_lower == "stability":
                    stable_features_list, _ = self.select_by_bootstrap_stability() # stability_df is ignored here
                    selected_features = stable_features_list
                else:
                    self.logger.error(f"Unsupported auto_k_selection_method: '{auto_k_selection_method}'. "
                                      "Choose 'coverage' or 'stability'.")
                    return []
            elif k is not None and k > 0:
                self.logger.info(f"--- Fixed-k Feature Selection (Target k={k}) ---")
                self.feature_scores.clear()

                current_numeric_features = self.numeric_cols[:]
                if 'variance' in self.config.methods_to_run or self.config.variance_threshold > 0:
                    current_numeric_features = self.filter_by_variance() # Uses self.config

                if not current_numeric_features:
                    self.logger.warning("No numeric features remaining after variance filtering (or none to begin with).")
                    return []

                if 'laplacian' in self.config.methods_to_run:
                    self.calculate_laplacian_score(features=current_numeric_features) # Uses self.config
                if 'pseudo_label' in self.config.methods_to_run:
                    self.calculate_ensemble_pseudo_label_scores(features=current_numeric_features) # Uses self.config

                features_that_got_scored = [
                    f for f in current_numeric_features
                    if f in self.feature_scores and (
                        self.feature_scores[f] or
                        ('variance' in self.feature_scores.get(f,{}) and 'variance' in self.config.methods_to_run)
                    )
                ]

                if not features_that_got_scored:
                    self.logger.warning("No features were scored by the enabled methods. "
                                        "Returning the top k from variance-filtered list if available, or empty.")
                    return current_numeric_features[:k]

                # Scope feature_scores for aggregation to only include currently relevant features
                # This is important if select() is called multiple times with different feature subsets
                scoped_feature_scores = defaultdict(dict, {
                    f: self.feature_scores.get(f, {}) for f in features_that_got_scored
                })
                original_scores_backup = self.feature_scores # Backup the main scores
                self.feature_scores = scoped_feature_scores # Use scoped for aggregation

                aggregated_df = self.aggregate_scores() # Uses self.config implicitly via score types

                self.feature_scores = original_scores_backup # Restore main scores

                if aggregated_df.empty:
                    self.logger.warning("Score aggregation resulted in an empty DataFrame. "
                                        "Returning the top k from variance-filtered list if available.")
                    return current_numeric_features[:k]

                pruned_df = aggregated_df
                if self.config.correlation_threshold is not None and self.config.correlation_threshold < 1.0:
                    # Use self.df (cleaned main df) for correlation data in fixed-k selection
                    pruned_df = self._prune_highly_correlated_features(aggregated_df, self.df) # Uses self.config

                if pruned_df.empty:
                    self.logger.warning("No features remaining after correlation pruning.")
                    return []

                final_k_to_select = min(k, len(pruned_df))
                selected_features = pruned_df['feature'].head(final_k_to_select).tolist()
                self.logger.info(
                    f"--- Feature Selection Completed (fixed-k). Selected {len(selected_features)} features: {selected_features[:20]}{'...' if len(selected_features)>20 else ''} ---"
                )
            else:
                self.logger.error("Invalid selection mode: Provide a positive integer 'k' or a "
                                  "valid 'auto_k_selection_method' ('coverage', 'stability').")
                return []

        finally:
            self.config = _original_instance_config # Restore original config

        return selected_features


    def select_k_by_coverage(self) -> List[str]:
        """Selects features based on achieving a cumulative importance coverage."""
        # This method uses self.config directly, which should be the (potentially temporarily overridden) config.
        self.logger.info(
            f"--- Automatic Feature Selection by Coverage (Target: {self.config.coverage_threshold*100:.1f}%) ---"
        )
        self.feature_scores.clear()

        current_numeric_features = self.numeric_cols[:]
        if 'variance' in self.config.methods_to_run or self.config.variance_threshold > 0:
            current_numeric_features = self.filter_by_variance()

        if not current_numeric_features:
            self.logger.warning("No numeric features remaining after variance filtering for coverage selection.")
            return []

        if 'laplacian' in self.config.methods_to_run:
            self.calculate_laplacian_score(features=current_numeric_features)
        if 'pseudo_label' in self.config.methods_to_run:
            self.calculate_ensemble_pseudo_label_scores(features=current_numeric_features)

        features_that_got_scored = [
            f for f in current_numeric_features
            if f in self.feature_scores and (
                self.feature_scores[f] or
                ('variance' in self.feature_scores.get(f,{}) and 'variance' in self.config.methods_to_run)
            )
        ]
        if not features_that_got_scored:
            self.logger.warning("No features were scored. Cannot select by coverage. Returning all variance-filtered features.")
            return current_numeric_features

        scoped_feature_scores = defaultdict(dict, {f: self.feature_scores.get(f, {}) for f in features_that_got_scored})
        original_scores_backup = self.feature_scores
        self.feature_scores = scoped_feature_scores

        aggregated_df = self.aggregate_scores()
        self.feature_scores = original_scores_backup

        if aggregated_df.empty or 'importance_score' not in aggregated_df.columns:
            self.logger.warning("Score aggregation failed or 'importance_score' missing. "
                                "Returning all variance-filtered features.")
            return current_numeric_features

        pruned_df = aggregated_df
        if self.config.correlation_threshold is not None and self.config.correlation_threshold < 1.0:
            pruned_df = self._prune_highly_correlated_features(aggregated_df, self.df) # Use self.df for correlation data

        if pruned_df.empty or 'importance_score' not in pruned_df.columns:
            self.logger.warning("No features remaining after correlation pruning or 'importance_score' missing.")
            return []

        features_sorted_by_importance = pruned_df.sort_values('importance_score', ascending=False)
        features_sorted_by_importance.dropna(subset=['importance_score'], inplace=True)
        features_sorted_by_importance = features_sorted_by_importance[np.isfinite(features_sorted_by_importance['importance_score'])]

        if features_sorted_by_importance.empty:
            self.logger.warning("No features with finite importance scores available for coverage selection.")
            return []

        importance_scores = features_sorted_by_importance['importance_score'].values
        if np.any(importance_scores < 0):
            self.logger.debug("Normalizing importance scores to be non-negative for coverage calculation.")
            importance_scores = importance_scores - np.min(importance_scores)

        total_importance = np.sum(importance_scores)
        if total_importance <= 1e-9:
            self.logger.warning("Total importance of features is near zero. "
                                "Returning all features sorted by importance (up to original number).")
            return features_sorted_by_importance['feature'].tolist()

        cumulative_importance_ratio = np.cumsum(importance_scores) / total_importance
        k_auto_coverage = np.searchsorted(cumulative_importance_ratio, self.config.coverage_threshold, side='left') + 1
        k_auto_coverage = min(k_auto_coverage, len(features_sorted_by_importance))

        selected_features = features_sorted_by_importance['feature'].head(k_auto_coverage).tolist()
        final_coverage_achieved = cumulative_importance_ratio[k_auto_coverage-1] if k_auto_coverage > 0 else 0.0
        self.logger.info(
            f"Selected {len(selected_features)} features by coverage "
            f"(achieved ~{final_coverage_achieved*100:.1f}%): {selected_features[:20]}{'...' if len(selected_features)>20 else ''}"
        )
        return selected_features


    def select_by_bootstrap_stability(self) -> Tuple[List[str], pd.DataFrame]:
        """
        Selects features based on their stability (frequency of selection)
        across multiple bootstrap samples of the original DataFrame.
        Each bootstrap sample undergoes a fixed-k selection process using this instance's current config.
        """
        # This method uses self.config directly.
        cfg = self.config
        self.logger.info(
            f"--- Automatic Feature Selection by Bootstrap Stability (B={cfg.bootstrap_B}, k_per_iter={cfg.bootstrap_k_per_iter}) ---"
        )

        if self.original_df_for_bootstrap.empty: # Use the pristine original df
            self.logger.error("Original DataFrame for bootstrapping is empty. Cannot perform stability selection.")
            return [], pd.DataFrame()

        feature_selection_counts = defaultdict(int)
        all_features_selected_across_bootstraps = set()

        for i in range(cfg.bootstrap_B):
            self.logger.info(f"Bootstrap iteration {i + 1}/{cfg.bootstrap_B}")

            bootstrap_rs = (self.random_state + i) if self.random_state is not None else None
            bootstrap_df = resample(
                self.original_df_for_bootstrap, # Sample from the original unmodified DataFrame
                n_samples=len(self.original_df_for_bootstrap),
                replace=True,
                random_state=bootstrap_rs
            )

            # Create a temporary FeatureSelector for this bootstrap sample.
            # It inherits the current main config (self.config, which might be overridden for the main select call)
            # but with verbose=False to keep logs clean.
            temp_selector_config_dict = vars(self.config).copy()
            # temp_selector_config_dict['verbose'] = False # This would affect the logger level, not just info messages

            temp_selector = FeatureSelector(
                df=bootstrap_df,
                id_cols=self.id_cols,
                target_col=self.target_col,
                random_state=bootstrap_rs,
                verbose=False, # Make the temporary selector non-verbose for its own logger
                n_jobs=self.n_jobs,
                config=FeatureSelectorConfig(**temp_selector_config_dict)
            )

            try:
                # Perform fixed-k selection on the bootstrap sample using the temp_selector's config
                bootstrap_selected_features = temp_selector.select(
                    k=cfg.bootstrap_k_per_iter, # Use k_per_iter from the main config
                    auto_k_selection_method=None # Crucial: force fixed-k selection
                )
                for feature in bootstrap_selected_features:
                    feature_selection_counts[feature] += 1
                    all_features_selected_across_bootstraps.add(feature)
            except Exception as e_boot:
                self.logger.warning(f"Bootstrap iteration {i + 1} failed during feature selection: {e_boot}.", exc_info=False)

        if not feature_selection_counts:
            self.logger.warning("No features were selected in any bootstrap iteration.")
            return [], pd.DataFrame()

        frequency_df = pd.DataFrame({
            'feature': list(all_features_selected_across_bootstraps),
            'frequency': [feature_selection_counts.get(f, 0) for f in all_features_selected_across_bootstraps]
        })
        frequency_df['selection_percentage'] = (frequency_df['frequency'] / cfg.bootstrap_B) * 100.0
        frequency_df.sort_values('frequency', ascending=False, inplace=True)
        frequency_df.reset_index(drop=True, inplace=True)

        self.logger.info(f"Bootstrap stability results (Top 20 features or all if fewer):\n{frequency_df.head(20)}")

        stable_features_list: List[str]
        if cfg.bootstrap_selection_freq_threshold is not None:
            min_frequency_value = cfg.bootstrap_selection_freq_threshold * cfg.bootstrap_B
            stable_features_df = frequency_df[frequency_df['frequency'] >= min_frequency_value]
            stable_features_list = stable_features_df['feature'].tolist()
            self.logger.info(
                f"Selected {len(stable_features_list)} features appearing in at least "
                f"{cfg.bootstrap_selection_freq_threshold*100:.1f}% of bootstrap iterations: "
                f"{stable_features_list[:20]}{'...' if len(stable_features_list)>20 else ''}"
            )
        else:
            self.logger.info("No bootstrap_selection_freq_threshold set. Returning all features ranked by selection frequency.")
            stable_features_list = frequency_df['feature'].tolist()

        return stable_features_list, frequency_df

