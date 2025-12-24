# -*- coding: utf-8 -*-
from probml.core.utils import get_logger
from probml.preprocessing.data_preprocessor import DataPreprocessor
from probml.preprocessing.sampling import SamplingUtils
from probml.analysis.dimensionality import DimensionalityReducer
from probml.analysis.clustering import ClusteringSuite
from probml.analysis.spatial import SpatialAnalyzer

# File: analysis_orchestrator.py
import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Any, Tuple, Union, Sequence
from collections import defaultdict
from joblib import Parallel, delayed, cpu_count
import pickle
import os

from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis


# --- Analysis Orchestrator Class ---

class AnalysisOrchestrator:
    """
    Orchestrates the robust real estate analysis pipeline.
    Uses ClusteringSuite for dynamic K selection and adapts clustering methods.
    """
    def __init__(self,
                 random_state: Optional[int] = None,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 max_total_time_seconds: Optional[int] = 3600,
                 price_col: str = 'sale_price',
                 log_price_col_name: str = 'log_sale_price',
                 apply_spatial_analysis: bool = False,
                 # Configs passed to DataPreprocessor.fit_transform
                 min_price_percentile: float = 0.005,
                 max_price_percentile: float = 0.995,
                 default_transform_type: Optional[str] = 'quantile',
                 transform_skew_threshold: float = 1.0,
                 # Orchestrator execution controls
                 max_samples_limit: int = 50000,
                 max_bootstraps: int = 30,
                 # Informational targets (not directly used for control)
                 target_dim_ci: float = 2.0,
                 target_k_ci: float = 1.0,
                 target_ari_ci: float = 0.05
                 ):

        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = cpu_count() if n_jobs <= 0 else n_jobs
        self.max_total_time_seconds = max_total_time_seconds
        self.logger = get_logger(self.__class__.__name__, verbose=self.verbose)

        self.price_col = price_col
        self.log_price_col = log_price_col_name

        # Store config params for potential use or passing down
        self.min_price_percentile = min_price_percentile
        self.max_price_percentile = max_price_percentile
        self.default_transform_type = default_transform_type
        self.transform_skew_threshold = abs(transform_skew_threshold)

        self.max_samples_limit = max_samples_limit
        self.max_bootstraps = max_bootstraps
        # Store informational targets
        self.target_dim_ci = target_dim_ci
        self.target_k_ci = target_k_ci
        self.target_ari_ci = target_ari_ci

        # Initialize components
        try:
            # Pass relevant general configs to components
            self.preprocessor = DataPreprocessor(
                price_col=self.price_col,
                log_price_col_name=self.log_price_col,
                random_state=self.random_state,
                verbose=self.verbose
                # Other DataPreprocessor configs are passed via fit_transform
            )
            self.sampler = SamplingUtils(random_state=self.random_state, verbose=self.verbose)
            self.dim_reducer = DimensionalityReducer(random_state=self.random_state, verbose=self.verbose, n_jobs=self.n_jobs)
            self.clusterer = ClusteringSuite(random_state=self.random_state, verbose=self.verbose, n_jobs=self.n_jobs)

            self.ica_module = ICAModule(random_state=self.random_state, verbose=self.verbose)
            self.logger.info("ICAModule initialized.")


        except NameError as ne:
            self.logger.error(f"Failed to initialize a component (NameError): {ne}. Ensure all dependent classes are defined/imported.", exc_info=True)
            raise RuntimeError(f"Orchestrator component initialization failed: {ne}")
        except Exception as e:
            self.logger.error(f"Unexpected error initializing components: {e}", exc_info=True)
            raise RuntimeError(f"Orchestrator component initialization failed: {e}")

        # Initialize Spatial Analyzer if requested and available
        self.apply_spatial_analysis = apply_spatial_analysis
        self.spatial_analyzer = None
        if self.apply_spatial_analysis:
            if GEOPANDAS_AVAILABLE and 'SpatialAnalyzer' in globals() and SpatialAnalyzer is not None:
                try:
                    self.spatial_analyzer = SpatialAnalyzer(random_state=self.random_state, verbose=self.verbose)
                    self.logger.info("SpatialAnalyzer initialized.")
                except NameError:
                     self.logger.warning("SpatialAnalyzer class not found despite GEOPANDAS_AVAILABLE=True.")
                     self.apply_spatial_analysis = False
            else:
                self.logger.warning("Spatial analysis requested but SpatialAnalyzer class or GeoPandas not available. Spatial steps will be skipped.")
                self.apply_spatial_analysis = False

        # State / Results storage
        self.results_history: Dict[str, Any] = defaultdict(dict)
        self.start_time_pipeline: Optional[float] = None
        self.fitted_price_model: Optional[Any] = None
        self.price_thresholds: Optional[List[float]] = None
        self._numeric_features_for_analysis: Optional[List[str]] = None
        self._max_samples_for_core_analysis: int = max_samples_limit
        self._bootstrap_params: Dict = {'run': False, 'n_samples': 0}

        self.logger.info("AnalysisOrchestrator initialized.")
        if self.verbose: check_available_methods(verbose=self.verbose) # Call utility if available

    def _check_timeout(self) -> bool:
        """Checks if the pipeline execution time has exceeded the maximum allowed."""
        if self.max_total_time_seconds and self.start_time_pipeline:
            elapsed = time.time() - self.start_time_pipeline
            if elapsed > self.max_total_time_seconds:
                self.logger.warning(f"Pipeline timeout ({self.max_total_time_seconds}s) reached (Elapsed: {elapsed:.1f}s). Stopping current stage.")
                return True
        return False

    def _get_numeric_data_and_features(self,
                                       df: pd.DataFrame,
                                       numeric_features_list: Optional[List[str]],
                                       context: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Extracts and cleans numeric data based on provided list or defaults."""
        self.logger.info(f"Preparing numeric data for {context}...")
        target_cols = numeric_features_list
        using_default_selection = False

        if not target_cols:
            self.logger.debug(f"No specific numeric features for {context}. Selecting defaults (all numeric excluding price/common IDs).")
            using_default_selection = True
            price_related = [self.price_col, self.log_price_col, getattr(self.preprocessor, 'price_outlier_col', 'is_price_outlier')]
            # Expanded list of potential ID-like column name fragments
            potential_id_fragments = [
                'borough', 'block', 'lot', 'zip', 'cd', 'council', 'dist', 'health', 'police', 'precinct',
                'year', 'date', 'bbl', 'coord', 'lat', 'lon', 'index', 'rowid', 'id', 'gid', 'fips', 'tract', 'bldg', 'unit',
                'census', 'nta', 'community', 'xcoord', 'ycoord', # Added more common ones
            ]
            target_cols = df.select_dtypes(include=np.number).columns.tolist()
            target_cols = [
                c for c in target_cols if c not in price_related and
                not any(pid_frag in c.lower() for pid_frag in potential_id_fragments)
            ]
            self.logger.debug(f"Default numeric features for {context}: {target_cols}")

        # Filter target_cols to those existing and numeric in df
        existing_cols = [col for col in target_cols if col in df.columns]
        final_numeric_features = []
        for col in existing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                final_numeric_features.append(col)
            elif not using_default_selection: # Warn only if a specific requested col is non-numeric
                self.logger.warning(f"Requested column '{col}' for {context} is not numeric. Skipping.")
        if len(existing_cols) < len(target_cols or []):
             missing = set(target_cols or []) - set(existing_cols)
             self.logger.warning(f"Requested columns missing from DataFrame for {context}: {missing}")


        if not final_numeric_features:
            self.logger.error(f"No valid numeric features found for {context}.")
            return None, None

        X_numeric = df[final_numeric_features].copy().astype(np.float64)

        # Robust NaN/Inf Cleaning
        if not np.all(np.isfinite(X_numeric.values)):
             cols_with_issues_before = X_numeric.columns[~np.all(np.isfinite(X_numeric.values), axis=0)].tolist()
             self.logger.debug(f"NaN/Inf detected in {len(cols_with_issues_before)} features for {context} before imputation: {cols_with_issues_before}")
             for col in final_numeric_features: # Iterate over the list of names
                 col_data_series = X_numeric[col]
                 if not np.all(np.isfinite(col_data_series.values)): # Check if this specific column needs cleaning
                     finite_vals = col_data_series[np.isfinite(col_data_series.values)]
                     if not finite_vals.empty:
                         median_val = finite_vals.median()
                         posinf_fill = finite_vals.max() if len(finite_vals) > 1 else median_val
                         neginf_fill = finite_vals.min() if len(finite_vals) > 1 else median_val
                         fill_val = median_val # For NaNs
                     else:
                         fill_val, posinf_fill, neginf_fill = 0.0, 0.0, 0.0
                         self.logger.warning(f"Column '{col}' for {context} is all NaN/Inf. Imputing with 0.")
                     X_numeric[col] = np.nan_to_num(col_data_series.values, nan=fill_val, posinf=posinf_fill, neginf=neginf_fill)

        if not np.all(np.isfinite(X_numeric.values)): # Final check
            cols_still_bad = X_numeric.columns[~np.all(np.isfinite(X_numeric.values), axis=0)].tolist()
            self.logger.error(f"Non-finite values persist after cleaning in {context} for cols: {cols_still_bad}. This may cause downstream errors.")
            return None, None

        if X_numeric.empty:
            self.logger.error(f"No numeric data remaining after processing for {context}.")
            return None, None

        self.logger.info(f"Numeric data for {context} prepared. Shape: {X_numeric.shape}, Features: {final_numeric_features}")
        return X_numeric.values, final_numeric_features

    def analyze_sample(self,
                       df_main_sample: pd.DataFrame,
                       numeric_features_for_analysis: Optional[List[str]],
                       full_analysis: bool = True,
                       # Dynamic K parameters for ensemble
                       ensemble_k_range: Sequence[int] = range(2, 12),
                       ensemble_k_metric: str = 'silhouette',
                       # DP-GMM config (now uses K* from find_optimal_k)
                       # dp_gmm_max_components_init_heuristic_factor: float = 2.0, # No longer needed
                       # dp_gmm_max_components_init_cap: int = 15, # No longer needed
                       dp_gmm_weight_prior: float = 0.1, # Keep other BGM params
                       # ICA config
                       use_ica_for_clustering: bool = False,
                       # Flag for bootstrap context
                       run_ensemble_k_selection_in_bootstrap: bool = False,
                       bootstrap_iter: Optional[int] = None) -> Dict[str, Any]:
        """Performs core analysis (Dim Est, Clustering) on a sample DataFrame."""
        sample_label = f"Iter {bootstrap_iter}" if bootstrap_iter is not None else "Main"
        self.logger.info(f"\n--- Analyzing Sample (Shape: {df_main_sample.shape}, Label: {sample_label}, FullAnalysis: {full_analysis}) ---")
        sample_artifacts: Dict[str, Any] = {'status': 'started', 'error': None, 'params': locals()}

        if self.clusterer is None or self.dim_reducer is None:
            msg = "ClusteringSuite or DimensionalityReducer not initialized."
            self.logger.error(msg); sample_artifacts.update({'status': 'error', 'error': msg}); return sample_artifacts

        try:
            X_numeric_values, actual_features = self._get_numeric_data_and_features(
                df_main_sample, numeric_features_for_analysis, f"sample_analysis_{sample_label}"
            )
            if X_numeric_values is None or not actual_features: raise ValueError("No valid numeric data.")

            # Remove constant columns before scaling
            variances = np.var(X_numeric_values, axis=0)
            constant_cols_mask = variances < 1e-12
            if np.any(constant_cols_mask):
                constant_feature_names = [name for i, name in enumerate(actual_features) if constant_cols_mask[i]]
                self.logger.warning(f"{sample_label}: Removing constant columns before scaling: {constant_feature_names}")
                X_numeric_values = X_numeric_values[:, ~constant_cols_mask]
                actual_features = [name for i, name in enumerate(actual_features) if not constant_cols_mask[i]]
                if X_numeric_values.shape[1] == 0: raise ValueError("All features became constant.")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric_values)
            if np.any(~np.isfinite(X_scaled)): # Safeguard
                self.logger.error(f"{sample_label}: NaNs/Infs DETECTED in X_scaled AFTER scaling. Imputing with 0.");
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            sample_artifacts.update({'core_features_used': actual_features, 'scaler_for_core_features': scaler, 'X_scaled_shape': X_scaled.shape})

            # --- Intrinsic Dimension ---
            if self._check_timeout(): raise TimeoutError("Timeout before Intrinsic Dimension")
            id_results = self.dim_reducer.estimate_intrinsic_dimension(X_scaled)
            sample_artifacts['intrinsic_dimension_results'] = id_results
            latent_dim_suggestion = id_results.get('median', max(1, X_scaled.shape[1] // 2))
            latent_dim_suggestion = max(1, min(int(latent_dim_suggestion), X_scaled.shape[1])) # Ensure valid dim
            sample_artifacts['latent_dim_suggestion'] = latent_dim_suggestion
            self.logger.info(f"{sample_label}: Suggested latent dim = {latent_dim_suggestion} (Method: {id_results.get('method', 'N/A')})")

            # --- Optional ICA ---
            data_for_clustering = X_scaled
            sample_artifacts['clustering_input_type'] = 'original_scaled_features'
            if self.ica_module and use_ica_for_clustering and latent_dim_suggestion > 0 and X_scaled.shape[1] >= latent_dim_suggestion:
                if self._check_timeout(): raise TimeoutError("Timeout during ICA")
                ica_n_components = min(latent_dim_suggestion, X_scaled.shape[1])
                # Assuming ica_module.run_ica returns a dict now
                ica_run_results = self.ica_module.run_ica(X_scaled, n_components=ica_n_components, feature_names=actual_features)
                sample_artifacts.update(ica_run_results) # Merge results
                ica_sources = ica_run_results.get('ica_sources')
                if ica_sources is not None and ica_sources.shape[1] > 0:
                    ica_scaler = StandardScaler(); data_for_clustering = ica_scaler.fit_transform(ica_sources)
                    sample_artifacts.update({'scaler_for_ica_sources': ica_scaler, 'clustering_input_type': 'scaled_ica_sources'})
                    self.logger.info(f"{sample_label}: Using {data_for_clustering.shape[1]} scaled ICA sources for clustering.")
                else: self.logger.warning(f"{sample_label}: ICA sources not generated or invalid. Using original scaled features.")

            if data_for_clustering.shape[0] <= 1: raise ValueError(f"Insufficient samples ({data_for_clustering.shape[0]}) for clustering.")

            # --- Dynamic K Selection (Common for DP-GMM target K and Ensemble) ---
            optimal_k_dynamic = 0
            # Run K-selection if it's the main analysis OR if specifically requested for bootstrap
            if full_analysis or run_ensemble_k_selection_in_bootstrap:
                if self._check_timeout(): raise TimeoutError("Timeout during Dynamic K Selection")
                self.logger.info(f"{sample_label}: Determining optimal K (Metric: {ensemble_k_metric})...")
                n_samples_for_k_search = data_for_clustering.shape[0]
                # Make k_range more robust against large N
                k_list = list(ensemble_k_range)
                max_k_possible = n_samples_for_k_search - 1 if n_samples_for_k_search > 1 else 1
                valid_k_range = [k for k in k_list if 1 < k <= max_k_possible]

                if not valid_k_range:
                    self.logger.warning(f"{sample_label}: Dynamic K range {k_list} invalid for sample size {n_samples_for_k_search}. Using fallback K.")
                    # Simple fallback: latent dim or 2
                    optimal_k_dynamic = min(max(2, latent_dim_suggestion), max_k_possible) if max_k_possible >=2 else 1
                else:
                    optimal_k_dynamic = self.clusterer.find_optimal_k(
                        X_processed=data_for_clustering, k_range=valid_k_range, metric=ensemble_k_metric
                    )
                sample_artifacts['optimal_k_dynamic'] = optimal_k_dynamic # Store the determined K*
                self.logger.info(f"{sample_label}: Optimal K dynamically determined as {optimal_k_dynamic}")
            else:
                # If not running K selection (e.g., bootstrap without the flag), use a reasonable default or placeholder
                 optimal_k_dynamic = min(max(2, latent_dim_suggestion), data_for_clustering.shape[0] - 1 if data_for_clustering.shape[0] > 1 else 1)
                 optimal_k_dynamic = max(1, optimal_k_dynamic) # Ensure >= 1
                 sample_artifacts['optimal_k_dynamic'] = optimal_k_dynamic # Store fallback K used
                 self.logger.info(f"{sample_label}: Using default/latent K = {optimal_k_dynamic} (K-selection skipped).")


            # --- DP-GMM Clustering (Using optimal_k_dynamic as target) ---
            if self._check_timeout(): raise TimeoutError("Timeout before DP-GMM")
            # Use the dynamically determined K as the target/upper bound for DP-GMM
            target_k_for_dp = optimal_k_dynamic

            dp_gmm_model, dp_gmm_info = self.clusterer.run_dp_gmm(
                data_for_clustering,
                target_n_components=target_k_for_dp,
                weight_concentration_prior=dp_gmm_weight_prior # Pass other relevant params
            )
            sample_artifacts.update({'dp_gmm_model_info_dict': dp_gmm_info})
            for key in ['effective_components', 'labels', 'model_converged', 'bic_score', 'aic_score']:
                artifact_key = f"dp_gmm_{key}" if key != 'effective_components' else 'dp_gmm_n_components'
                sample_artifacts[artifact_key] = dp_gmm_info.get(key)
            self.logger.info(f"{sample_label}: DP-GMM effective components = {sample_artifacts.get('dp_gmm_n_components', 'N/A')} (Target K={target_k_for_dp})")


            # --- Full Ensemble Clustering (only if full_analysis is True) ---
            # Use the *same* optimal_k_dynamic determined earlier
            if full_analysis and optimal_k_dynamic > 0:
                if self._check_timeout(): raise TimeoutError("Timeout during Ensemble Clustering")
                self.logger.info(f"{sample_label}: Running Ensemble Clustering with K={optimal_k_dynamic}...")
                # Pass the *data* used for clustering (could be original or ICA)
                ensemble_partitions, ensemble_consensus_info = self.clusterer.run_ensemble_clustering(
                    X=data_for_clustering, # Use the potentially ICA-transformed data
                    n_clusters=optimal_k_dynamic
                )
                sample_artifacts['ensemble_partitions_shapes'] = {k:v.shape for k,v in ensemble_partitions.items() if hasattr(v, 'shape')}
                sample_artifacts['ensemble_consensus_summary'] = {k:v for k,v in ensemble_consensus_info.items() if k not in ['co_association_matrix', 'final_labels']}
                sample_artifacts['ensemble_labels'] = ensemble_consensus_info.get('final_labels')
                sample_artifacts['ensemble_n_clusters'] = ensemble_consensus_info.get('n_clusters_found', optimal_k_dynamic)
                sample_artifacts['ensemble_silhouette_score'] = ensemble_consensus_info.get('silhouette_score')
            elif full_analysis:
                self.logger.warning(f"{sample_label}: Skipping ensemble clustering as optimal K was {optimal_k_dynamic}.")
                sample_artifacts['ensemble_clustering_status'] = f'skipped_k_{optimal_k_dynamic}'


            sample_artifacts['status'] = 'completed'
        except TimeoutError as te:
            sample_artifacts.update({'status': 'timeout', 'error': f'Timeout: {str(te)}'})
        except Exception as e:
            self.logger.error(f"Error during analyze_sample for {sample_label}: {e}", exc_info=True)
            sample_artifacts.update({'status': 'error', 'error': str(e)})

        self.logger.info(f"--- Sample Analysis ({sample_label}) Completed: {sample_artifacts['status']} ---")
        return sample_artifacts

    def full_analysis_pipeline(self,
                               df_input: pd.DataFrame,
                               numeric_features_for_analysis: Optional[List[str]] = None,
                               transform_numeric_features: Optional[List[str]] = None, # Which features to consider for skew transform
                               # Configs passed to DataPreprocessor.fit_transform:
                               transform_type: Optional[str] = None, # Specific transform to use if auto_transform_skewed=False
                               auto_transform_skewed: bool = True,   # Auto-detect and transform skewed features?
                               run_bootstrap: bool = True,
                               n_bootstrap_samples: int = 30,
                               max_samples_for_core_analysis: int = 20000,
                               # Configs passed to analyze_sample -> find_optimal_k:
                               ensemble_k_range: Sequence[int] = range(2, 12),
                               ensemble_k_metric: str = 'silhouette',
                               # Spatial params:
                               spatial_x_coord: Optional[str] = None,
                               spatial_y_coord: Optional[str] = None,
                               spatial_grid_size: Union[int, float] = 250,
                               spatial_crs: Optional[str] = None
                               ) -> Dict[str, Any]:
        """Runs the full pipeline: Preprocessing, Sampling, Analysis, Bootstrap, Summary."""
        self.start_time_pipeline = time.time()
        self.logger.info(f"====== Starting Full Analysis Pipeline (Bootstrap: {run_bootstrap}, MaxSamplesCore: {max_samples_for_core_analysis}) ======")
        # Store key parameters used for this run
        self._numeric_features_for_analysis = numeric_features_for_analysis
        self._max_samples_for_core_analysis = max_samples_for_core_analysis
        self._bootstrap_params = {'run': run_bootstrap, 'n_samples': n_bootstrap_samples}
        self.results_history = defaultdict(dict) # Reset history

        overall_results: Dict[str, Any] = {
            "summary_report": {}, "main_artifacts": {}, "bootstrap_summary": {},
            "status": "started", "error": None
        }
        main_artifacts: Dict[str, Any] = {} # To store results of main analysis run

        try:
            # --- Stage 1: Preprocessing ---
            self.logger.info("--- Stage 1: Preprocessing ---")
            # Determine which features DataPreprocessor should attempt to transform based on skew
            features_to_consider_for_transform = transform_numeric_features if transform_numeric_features is not None else numeric_features_for_analysis

            # Use orchestrator's stored config for price percentiles and thresholds
            df_processed = self.preprocessor.fit_transform(
                df=df_input.copy(),
                numeric_features_to_process=features_to_consider_for_transform, # Features DP should potentially transform/scale
                auto_transform_skewed=auto_transform_skewed,
                # Pass specific overrides relevant to DP's fit_transform method
                min_price_percentile=self.min_price_percentile,
                max_price_percentile=self.max_price_percentile,
                # Pass transform config only if auto_transform is enabled or a specific type is given
                default_transform_type_override=(self.default_transform_type if auto_transform_skewed else transform_type),
                transform_skew_threshold_override=(self.transform_skew_threshold if auto_transform_skewed else None),
                scale_numeric=True, # Assume we always scale numeric features after potential transform
                create_bbl=True   # Assume BBL creation is desired
            )
            if df_processed is None or df_processed.empty: raise ValueError("Preprocessing returned empty DataFrame.")
            self.results_history['preprocessing']['shape_after'] = df_processed.shape
            self.results_history['preprocessing']['price_stats'] = getattr(self.preprocessor, 'price_analysis_stats', {})
            overall_results['price_analysis_stats'] = getattr(self.preprocessor, 'price_analysis_stats', {})


            # --- Stage 2: Price Modeling and Banding ---
            self.logger.info("--- Stage 2: Price Modeling and Banding ---")
            self.fitted_price_model = None; self.price_thresholds = None
            if STUDENT_T_AVAILABLE and 'StudentTMixture' in globals() and StudentTMixture and self.log_price_col in df_processed.columns:
                price_data = df_processed[self.log_price_col].dropna().values.reshape(-1, 1)
                if len(price_data) > 10: # Only fit if enough data
                    try:
                        # Use a reasonable n_components, e.g., 3 for low/medium/high
                        price_model = StudentTMixture(n_components=3, random_state=self.random_state)
                        if price_data.shape[0] > price_model.n_components: # type: ignore
                            price_model.fit(price_data); self.fitted_price_model = price_model
                            if price_model.converged_ and price_model.means_ is not None and len(price_model.means_.flatten()) > 1:
                                sorted_means = sorted(price_model.means_.flatten())
                                self.price_thresholds = [(sorted_means[i] + sorted_means[i+1])/2.0 for i in range(len(sorted_means)-1)]
                            self.results_history['price_model_fit'] = {'converged': price_model.converged_, 'n_iter': price_model.n_iter_}
                    except Exception as e: self.logger.error(f"Price modeling (StudentT) failed: {e}", exc_info=False)

            if self.price_thresholds is None: # Fallback to quantiles
                self.logger.warning("Using quantile thresholds for price banding.")
                if self.log_price_col in df_processed.columns:
                     log_price_data_clean = df_processed[self.log_price_col].dropna()
                     if len(log_price_data_clean) > 0 :
                        self.price_thresholds = [log_price_data_clean.quantile(q) for q in [0.33, 0.67]]
                        self.price_thresholds = [t for t in self.price_thresholds if pd.notna(t)] # Remove potential NaNs
                     else: self.price_thresholds = []
                else: self.price_thresholds = []

            # Apply banding based on determined thresholds
            if self.price_thresholds and self.log_price_col in df_processed.columns:
                # Ensure bins are unique and sorted, including -inf and +inf
                bins = sorted(list(set([-np.inf] + self.price_thresholds + [np.inf])));
                if len(bins) >= 2 :
                    # Adjust labels based on number of bins
                    num_bands = len(bins) - 1
                    if num_bands == 1: labels = ['medium']
                    elif num_bands == 2: labels = ['low', 'high']
                    elif num_bands == 3: labels = ['low','medium','high']
                    else: labels = [f'band_{i}' for i in range(num_bands)]

                    try:
                        df_processed['price_band'] = pd.cut(df_processed[self.log_price_col], bins=bins, labels=labels, right=False, duplicates='drop')
                        # Handle potential NaNs created by cut if values fall outside explicit bins (shouldn't happen with inf)
                        # or if log_price_col itself was NaN
                        df_processed['price_band'] = df_processed['price_band'].cat.add_categories('unknown').fillna('unknown')

                    except ValueError as cut_err:
                        self.logger.error(f"Error pd.cut price_band: {cut_err}. Assigning 'unknown'.");
                        df_processed['price_band'] = 'unknown'
                    self.results_history['price_banding'] = {'bands_used': labels, 'thresholds_log': self.price_thresholds, 'bins': bins}
                else: # Only one bin possible (-inf, inf)
                     df_processed['price_band'] = 'medium'
                     self.results_history['price_banding'] = {'bands_used': ['medium'], 'thresholds_log': self.price_thresholds, 'bins': bins}
            else: # No thresholds determined or no log price col
                df_processed['price_band'] = 'unknown' # Use 'unknown' if banding fails
                self.results_history['price_banding'] = {'status': 'skipped_no_thresholds'}
            self.logger.info(f"Price banding applied. Bands found: {df_processed['price_band'].unique()}")


            # --- Stage 3: Rebalancing Bands ---
            self.logger.info("--- Stage 3: Rebalancing Bands ---")
            # If you need weights:
            df_rebalanced, sample_weights = self.sampler.rebalance_price_bands(
                df_processed,
                price_band_col='price_band',
                log_price_col=self.log_price_col,
                price_col=self.price_col,
                target_balance='median',
                downsample_overrepresented=True,
                return_weights=True
            )
            self.results_history['rebalancing']['shape_after'] = df_rebalanced.shape
            self.logger.info(f"Shape after rebalancing: {df_rebalanced.shape}")

            # --- Stage 4: Spatial Gridding (Optional) ---
            self.logger.info("--- Stage 4: Spatial Gridding (Optional) ---")
            df_final_for_sampling = df_rebalanced
            if self.apply_spatial_analysis and self.spatial_analyzer:
                if spatial_x_coord and spatial_y_coord and spatial_x_coord in df_final_for_sampling.columns and spatial_y_coord in df_final_for_sampling.columns:
                    if self._check_timeout(): raise TimeoutError("Timeout during Spatial Gridding")
                    df_final_for_sampling = self.spatial_analyzer.assign_grid_indices(df_rebalanced, spatial_x_coord, spatial_y_coord, spatial_grid_size, spatial_crs)
                    self.results_history['spatial_gridding'] = getattr(self.spatial_analyzer, 'grid_params', {'status':'completed'})
                else:
                    self.logger.warning("Spatial analysis requested but coordinate columns invalid/missing. Skipping gridding.")
                    self.results_history['spatial_gridding'] = {'status':'skipped_missing_coords'}
            else: self.results_history['spatial_gridding'] = {'status':'skipped_disabled'}
            self.results_history['df_fully_processed_before_sampling_shape'] = df_final_for_sampling.shape


            # --- Stage 5: Downsampling for Core Analysis ---
            self.logger.info(f"--- Stage 5: Downsampling for Core Analysis (Target: {max_samples_for_core_analysis}) ---")
            actual_core_sample_size = min(max_samples_for_core_analysis, len(df_final_for_sampling))
            if actual_core_sample_size <= 0: raise ValueError("Dataset empty before core downsampling.")
            # Use stratified sampling if price_band exists and has >1 category, else random
            stratify_col_core = 'price_band' if 'price_band' in df_final_for_sampling.columns and df_final_for_sampling['price_band'].nunique() > 1 else None
            self.logger.info(f"Downsampling to {actual_core_sample_size} for core analysis. Stratify by: {stratify_col_core}")
            df_main_sample, _ = self.sampler.downsample(
                df_final_for_sampling, n_samples=actual_core_sample_size,
                stratify_col=stratify_col_core,
                replace=False
            )
            if df_main_sample.empty: raise ValueError("Main sample is empty after downsampling.")
            self.results_history['main_sampling'] = {'sample_shape': df_main_sample.shape, 'stratified_by': stratify_col_core}


            # --- Stage 6: Main Sample Analysis ---
            self.logger.info("--- Stage 6: Main Sample Analysis ---")
            if self._check_timeout(): raise TimeoutError("Timeout before Main Analysis")
            main_artifacts = self.analyze_sample(
                df_main_sample=df_main_sample,
                numeric_features_for_analysis=self._numeric_features_for_analysis,
                full_analysis=True, # Run full ensemble clustering on main sample
                ensemble_k_range=ensemble_k_range,
                ensemble_k_metric=ensemble_k_metric,
                # Don't need run_ensemble_k_selection_in_bootstrap=True here, it's not bootstrap
            )
            # Add price stats from preprocessing to main artifacts for context
            main_artifacts['price_analysis_stats'] = self.results_history.get('preprocessing',{}).get('price_stats',{})
            overall_results["main_artifacts"] = main_artifacts
            if main_artifacts.get("status") != 'completed': raise ValueError(f"Main analysis failed: {main_artifacts.get('error', 'Unknown error')}")


            # --- Stage 7: Bootstrap Analysis ---
            bootstrap_aggregated_stats = {}
            if run_bootstrap and not self._check_timeout():
                n_boot = min(n_bootstrap_samples, self.max_bootstraps) # Use the smaller value
                self.logger.info(f"--- Stage 7: Bootstrap Analysis ({n_boot} samples) ---")
                # Bootstrap sample size should match the main analysis sample size
                bootstrap_sample_size_for_iter = actual_core_sample_size
                if bootstrap_sample_size_for_iter <= 0:
                     self.logger.warning("Main sample size 0, cannot run bootstrap.")
                     self.results_history['bootstrap_analysis_status'] = 'skipped_zero_main_sample'
                else:
                    # Stratify bootstrap sampling if possible, using the same column as main sampling
                    stratify_col_boot = stratify_col_core

                    # Define the function to run in parallel for one bootstrap iteration
                    def run_one_bootstrap_iteration(iter_num: int):
                        # Check timeout at start of worker task
                        if self._check_timeout(): return {'status': 'timeout', 'error': 'timeout in bootstrap worker', 'iter': iter_num}
                        try:
                            # Sample WITH REPLACEMENT from the *final_for_sampling* pool
                            boot_df, _ = self.sampler.downsample(
                                df_final_for_sampling,
                                n_samples=bootstrap_sample_size_for_iter,
                                stratify_col=stratify_col_boot,
                                replace=True # <<< Bootstrap uses replacement
                            )
                            if boot_df.empty: return {'status': 'error', 'error': 'empty_bootstrap_sample', 'iter': iter_num}

                            # Analyze the bootstrap sample:
                            # full_analysis=False (don't need full ensemble results, just K)
                            # run_ensemble_k_selection_in_bootstrap=True (need optimal K metric)
                            return self.analyze_sample(
                                df_main_sample=boot_df,
                                numeric_features_for_analysis=self._numeric_features_for_analysis,
                                full_analysis=False, # Don't need full ensemble clustering results
                                run_ensemble_k_selection_in_bootstrap=True, # DO run K-selection
                                ensemble_k_range=ensemble_k_range, # Use same range/metric
                                ensemble_k_metric=ensemble_k_metric,
                                bootstrap_iter=iter_num
                            )
                        except Exception as boot_e_inner:
                            # Log error from worker if possible
                            worker_logger = get_logger(f"{self.__class__.__name__}.BootstrapWorker", verbose=self.verbose)
                            worker_logger.error(f"Error in bootstrap worker {iter_num}: {boot_e_inner}", exc_info=False)
                            return {'status': 'error', 'error': f'Exception in worker {iter_num}: {str(boot_e_inner)}', 'iter': iter_num}

                    # Run bootstrap iterations in parallel
                    try:
                        self.logger.info(f"Running {n_boot} bootstrap iterations using {self.n_jobs} jobs...")
                        parallel_verbose_level = 5 if self.verbose else 0
                        bootstrap_results_list = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=parallel_verbose_level)(
                            delayed(run_one_bootstrap_iteration)(i) for i in range(n_boot)
                        )

                        # Process results
                        valid_bootstrap_results = [r for r in bootstrap_results_list if isinstance(r, dict) and r.get('status') == 'completed']
                        failed_runs = [r for r in bootstrap_results_list if not (isinstance(r, dict) and r.get('status') == 'completed')]

                        self.results_history['bootstrap_runs_raw'] = bootstrap_results_list # Store raw results for debugging

                        if valid_bootstrap_results:
                            bootstrap_aggregated_stats = self._aggregate_bootstrap_results(valid_bootstrap_results)
                            self.results_history['bootstrap_summary'] = bootstrap_aggregated_stats
                            # Refine main artifacts using bootstrap stats (e.g., update suggested K)
                            self._refine_artifacts_with_bootstrap(main_artifacts, bootstrap_aggregated_stats)
                            overall_results["main_artifacts"] = main_artifacts # Update with refined versions
                            self.logger.info(f"Completed {len(valid_bootstrap_results)} successful bootstrap runs.")
                        else:
                            self.logger.warning("No successful bootstrap runs completed.")

                        if failed_runs:
                            self.logger.warning(f"{len(failed_runs)} bootstrap runs failed or timed out.")
                            # Log details of first few failures
                            for i, failed_run in enumerate(failed_runs[:min(3, len(failed_runs))]):
                                self.logger.warning(f"  Failure Detail {i+1}: {failed_run}")

                        self.results_history['bootstrap_analysis_status'] = f'completed_{len(valid_bootstrap_results)}_ok_{len(failed_runs)}_failed'

                    except Exception as e_par:
                        self.logger.error(f"Bootstrap parallel execution failed: {e_par}", exc_info=True)
                        self.results_history['bootstrap_analysis_status'] = f'error_{str(e_par)}'

            else: # Bootstrap skipped
                 self.results_history['bootstrap_analysis_status'] = 'skipped_due_to_config_or_timeout'


            # --- Stage 8: Final Summary ---
            self.logger.info("--- Stage 8: Generating Final Summary ---")
            final_summary = self._generate_pipeline_summary(main_artifacts, bootstrap_aggregated_stats)
            overall_results.update({
                "summary_report": final_summary,
                "status": "completed",
                # expose the fully-processed, rebalanced DataFrame for downstream steps
                "df_processed_and_rebalanced": df_final_for_sampling
            })

        except TimeoutError as te_pipe:
             self.logger.error(f"Pipeline stopped due to timeout: {te_pipe}", exc_info=False)
             overall_results.update({"error": f"Timeout Error: {te_pipe}", "status": "timeout"})
             overall_results["main_artifacts"] = main_artifacts # Include whatever main artifacts were generated

        except Exception as e_pipe:
            self.logger.error(f"FATAL Error in full analysis pipeline: {e_pipe}", exc_info=True)
            overall_results.update({"error": str(e_pipe), "status": "error"})
            overall_results["main_artifacts"] = main_artifacts # Include partial artifacts if available

        total_time_taken = time.time() - self.start_time_pipeline
        self.logger.info(f"====== Pipeline Completed in {total_time_taken:.2f}s (Status: {overall_results['status']}) ======")
        overall_results['total_time_seconds'] = total_time_taken
        overall_results['detailed_results_history'] = dict(self.results_history) # Convert defaultdict for saving

        return overall_results

    def _aggregate_bootstrap_results(self, bootstrap_results: List[Dict]) -> Dict:
        """Aggregates statistics from multiple bootstrap runs."""
        if not bootstrap_results: return {}
        self.logger.info(f"Aggregating results from {len(bootstrap_results)} bootstrap runs...")
        agg_stats = defaultdict(list) # Use defaultdict to easily append

        # Keys to aggregate and their artifact key name
        keys_to_aggregate = {
            'latent_dim_suggestion': 'intrinsic_dim',
            'dp_gmm_n_components': 'dp_gmm_components',
            'optimal_k_dynamic': 'ensemble_optimal_k', # K chosen by find_optimal_k
            # Add other numeric metrics if needed, e.g., 'ensemble_silhouette_score' if calculated
        }

        for run_result in bootstrap_results:
            for artifact_key, base_key in keys_to_aggregate.items():
                 value = run_result.get(artifact_key)
                 if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                     agg_stats[base_key].append(value)

        final_agg_stats = {}
        for base_key, values in agg_stats.items():
             if values:
                 final_agg_stats[f'{base_key}_mean'] = float(np.mean(values))
                 final_agg_stats[f'{base_key}_median'] = float(np.median(values))
                 final_agg_stats[f'{base_key}_std'] = float(np.std(values))
                 q25, q75 = np.percentile(values, [25, 75])
                 final_agg_stats[f'{base_key}_q25'] = float(q25)
                 final_agg_stats[f'{base_key}_q75'] = float(q75)
                 final_agg_stats[f'{base_key}_count'] = len(values)

        self.logger.info(f"Aggregated bootstrap stats: {final_agg_stats}")
        return final_agg_stats

    def _refine_artifacts_with_bootstrap(self, main_artifacts: Dict, bootstrap_stats: Dict):
        """Refines estimates in main_artifacts using stable results from bootstrap."""
        if not bootstrap_stats or not main_artifacts: return
        self.logger.info("Refining main artifacts using bootstrap statistics...")

        # Refine Intrinsic Dimension
        if 'intrinsic_dim_median' in bootstrap_stats:
            new_dim = int(round(bootstrap_stats['intrinsic_dim_median'])); new_dim = max(1, new_dim)
            old_dim = main_artifacts.get('latent_dim_suggestion')
            if old_dim is None or new_dim != old_dim:
                self.logger.info(f"Refining latent_dim_suggestion from {old_dim} to {new_dim} based on bootstrap median.")
                main_artifacts['latent_dim_suggestion'] = new_dim
                if 'intrinsic_dimension_results' in main_artifacts:
                    main_artifacts['intrinsic_dimension_results']['median_refined_bootstrap'] = new_dim

        # Refine DP-GMM K suggestion (although DP-GMM finds effective K anyway)
        if 'dp_gmm_components_median' in bootstrap_stats:
            new_k = int(round(bootstrap_stats['dp_gmm_components_median'])); new_k = max(1, new_k)
            old_k = main_artifacts.get('dp_gmm_n_components') # This was the *effective* K from main run
            self.logger.info(f"Bootstrap median for DP-GMM effective components is {new_k} (main run found {old_k}). Storing for reference.")
            main_artifacts['dp_gmm_n_components_bootstrap_median'] = new_k
            # We might choose NOT to override the main run's effective K, as DP-GMM adapts.

        # Refine Optimal K for Ensemble
        if 'ensemble_optimal_k_median' in bootstrap_stats:
            new_ensemble_k = int(round(bootstrap_stats['ensemble_optimal_k_median'])); new_ensemble_k = max(1, new_ensemble_k)
            old_ensemble_k = main_artifacts.get('optimal_k_dynamic') # K chosen in main run
            if old_ensemble_k is None or new_ensemble_k != old_ensemble_k:
                self.logger.info(f"Refining optimal_k_dynamic from {old_ensemble_k} to {new_ensemble_k} based on bootstrap median.")
                main_artifacts['optimal_k_dynamic'] = new_ensemble_k # Override the dynamically chosen K
            main_artifacts['optimal_k_ensemble_bootstrap_median'] = new_ensemble_k # Store for reference regardless

    def _generate_pipeline_summary(self, main_artifacts: Dict, bootstrap_stats: Dict) -> Dict:
        """Generates a concise summary report from main and bootstrap results."""
        summary = {}
        if not main_artifacts: return {"status": "error - no main artifacts"}

        # Intrinsic Dimension Summary
        id_res = main_artifacts.get('intrinsic_dimension_results', {})
        # Use the refined dimension if available
        dim_est = main_artifacts.get('latent_dim_suggestion', id_res.get('median', 'N/A'))
        dim_method = id_res.get('method', 'N/A')
        dim_std_bs = bootstrap_stats.get('intrinsic_dim_std')
        summary['intrinsic_dimension'] = {
            'point_estimate': dim_est, 'method': dim_method,
            'bootstrap_median': bootstrap_stats.get('intrinsic_dim_median'),
            'bootstrap_std': f"{dim_std_bs:.2f}" if dim_std_bs is not None else 'N/A',
            'bootstrap_q25': bootstrap_stats.get('intrinsic_dim_q25'),
            'bootstrap_q75': bootstrap_stats.get('intrinsic_dim_q75'),
        }

        # Clustering Summary
        dp_k_main = main_artifacts.get('dp_gmm_n_components', 'N/A') # Effective K from main run
        dp_k_bs_median = bootstrap_stats.get('dp_gmm_components_median') # Median effective K from bootstrap
        dp_k_bs_std = bootstrap_stats.get('dp_gmm_components_std')

        ens_k_chosen_main = main_artifacts.get('optimal_k_dynamic', 'N/A') # K chosen by find_optimal_k (possibly refined)
        ens_k_final_run = main_artifacts.get('ensemble_n_clusters') # Actual K after ensemble ran (if different)
        ens_k_bs_median = bootstrap_stats.get('ensemble_optimal_k_median') # Median K chosen by find_optimal_k in bootstrap
        ens_k_bs_std = bootstrap_stats.get('ensemble_optimal_k_std')

        summary['clustering'] = {
            'dp_gmm_effective_k_main': dp_k_main,
            'dp_gmm_bootstrap_median_k': f"{dp_k_bs_median:.1f}" if dp_k_bs_median is not None else 'N/A',
            'dp_gmm_bootstrap_std_k': f"{dp_k_bs_std:.2f}" if dp_k_bs_std is not None else 'N/A',
            'ensemble_k_chosen': ens_k_chosen_main,
            'ensemble_k_final_clusters': ens_k_final_run if ens_k_final_run != ens_k_chosen_main else None, # Only show if different
            'ensemble_k_bootstrap_median': f"{ens_k_bs_median:.1f}" if ens_k_bs_median is not None else 'N/A',
            'ensemble_k_bootstrap_std': f"{ens_k_bs_std:.2f}" if ens_k_bs_std is not None else 'N/A',
            'ensemble_silhouette_main_run': f"{main_artifacts.get('ensemble_silhouette_score'):.3f}" if isinstance(main_artifacts.get('ensemble_silhouette_score'), float) else 'N/A',
        }

        summary['vae_recommendations'] = self._generate_vae_recommendations(main_artifacts) # Uses refined K
        summary['price_analysis'] = main_artifacts.get('price_analysis_stats', {})
        summary['core_features_used'] = main_artifacts.get('core_features_used', [])[:15] # Show first few
        summary['clustering_input_type'] = main_artifacts.get('clustering_input_type', 'unknown')

        return summary

    def _generate_vae_recommendations(self, main_artifacts: Dict) -> List[str]:
        """Generates VAE architecture and prior recommendations based on analysis."""
        recs = []
        # Use the potentially refined latent dimension suggestion
        latent_dim = main_artifacts.get('latent_dim_suggestion', 2)
        latent_dim = max(1, int(latent_dim)) # Ensure positive integer

        # Use the potentially refined ensemble K as the primary indicator for mixtures
        n_mixtures = main_artifacts.get('optimal_k_dynamic', # Refined ensemble K
                                     main_artifacts.get('dp_gmm_n_components', 1)) # Fallback to DP-GMM effective K
        n_mixtures = int(round(max(1, n_mixtures if isinstance(n_mixtures, (int, float)) else 1)))

        # Estimate input dim for VAE (based on features used for clustering + potentially price)
        core_features = main_artifacts.get('core_features_used', [])
        input_dim_vae_placeholder = len(core_features)
        # Assume log_price might be added as an input feature to VAE if not already in core_features
        if self.log_price_col not in core_features:
             input_dim_vae_placeholder += 1
        input_dim_vae_placeholder = max(1, input_dim_vae_placeholder)


        # Simple heuristic for hidden layers
        h1_dim = max(latent_dim * 2, input_dim_vae_placeholder // 3, latent_dim + 5) # Ensure hidden > latent
        h1_dim = min(h1_dim, input_dim_vae_placeholder * 2) # Avoid excessive expansion
        h1_dim = min(h1_dim, 512) # Cap max hidden size
        h1_dim = max(h1_dim, latent_dim + 1 if input_dim_vae_placeholder > latent_dim else latent_dim) # Must be >= latent_dim

        h2_dim = max(int(h1_dim * 0.5), latent_dim + 1 if h1_dim > latent_dim + 1 else latent_dim) # Second layer smaller

        hidden_layers_enc = []
        if h1_dim > latent_dim : hidden_layers_enc.append(h1_dim)
        if h2_dim > latent_dim and h2_dim < h1_dim: hidden_layers_enc.append(h2_dim)

        arch_str_enc = str(hidden_layers_enc) if hidden_layers_enc else "[direct]"
        hidden_layers_dec = hidden_layers_enc[::-1]
        arch_str_dec = str(hidden_layers_dec) if hidden_layers_dec else "[direct]"

        recs.append(f"Suggest VAE Arch: Input({input_dim_vae_placeholder}) -> Enc{arch_str_enc} -> Latent({latent_dim}) -> Dec{arch_str_dec} -> Output({input_dim_vae_placeholder}).")

        # Prior Type Recommendation
        prior_type_rec = "Gaussian"
        price_heavy_tailed = main_artifacts.get('price_analysis_stats', {}).get('log_heavy_tailed', False)
        # Add check for latent kurtosis if ICA stats are collected and stored under 'latent_dim_stats'
        latent_heavy_tailed = False
        if 'latent_dim_stats' in main_artifacts:
             kurtosis_values = [v.get('kurtosis',0) for k,v in main_artifacts.get('latent_dim_stats', {}).items() if isinstance(v, dict)]
             if kurtosis_values: latent_heavy_tailed = any(k > 1.0 for k in kurtosis_values)

        if price_heavy_tailed or latent_heavy_tailed: prior_type_rec = "Student-T"

        final_prior_rec = prior_type_rec + (" Mixture" if n_mixtures > 1 else "")
        recs.append(f"Use {final_prior_rec} prior (K={n_mixtures}, df~4 if Student-T) based on data/latent distributions.")

        # General VAE training recommendations
        recs.append("Consider learnable per-feature reconstruction variance (Gaussian NLL) or Huber loss for robustness.")
        recs.append("Use KL annealing (e.g., ~20-50 epochs or 25-50% of total training duration).")
        recs.append("Employ gradient clipping (e.g., max_norm=1.0) and LR scheduling (e.g., ReduceLROnPlateau).")
        if input_dim_vae_placeholder > 50 : recs.append("For high input dims, consider more aggressive bottleneck or deeper encoder/decoder.")

        return recs

