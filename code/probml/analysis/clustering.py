# -*- coding: utf-8 -*-
from probml.core.utils import get_logger, HDBSCAN_AVAILABLE

import numpy as np
import pandas as pd
import warnings
import time
import logging
from typing import Tuple, Optional, Dict, Any, List, Union, Sequence
import itertools

# Scikit-learn imports
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import resample as sk_resample # For subsampling in k_search


CLUSTERING_SUITE_AVAILABLE = True # Assuming this class itself means the suite is available

class ClusteringSuite:
    """
    Provides a suite of clustering algorithms and utilities, including DP-GMM,
    ensemble clustering, and methods for finding an optimal number of clusters (K).
    Handles preprocessing, optional dependencies, and performance considerations.
    """
    def __init__(self,
                 random_state: Optional[int] = None,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 use_hdbscan_if_available: bool = True,
                 default_sample_size_for_k_search: int = 5000,
                 ensemble_agglo_sample_threshold: int = 3000,
                 consensus_spectral_threshold: int = 2000
                 ):
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.use_hdbscan_if_available = use_hdbscan_if_available and HDBSCAN_AVAILABLE
        self.default_sample_size_for_k_search = default_sample_size_for_k_search
        self.ensemble_agglo_sample_threshold = ensemble_agglo_sample_threshold
        self.consensus_spectral_threshold = consensus_spectral_threshold

        self.last_dp_gmm_model: Optional[BayesianGaussianMixture] = None
        self.last_dp_gmm_labels: Optional[np.ndarray] = None

        self.class_logger = get_logger(f"{self.__class__.__name__}", verbose=self.verbose)
        self.class_logger.info(f"ClusteringSuite initialized. HDBSCAN Available: {HDBSCAN_AVAILABLE}, Will Use: {self.use_hdbscan_if_available}")

    def _log(self, message: str, level: str = "info", exc_info: bool = False):
        if self.verbose:
            if level == "info":    self.class_logger.info(message)
            elif level == "warning": self.class_logger.warning(message)
            elif level == "error":   self.class_logger.error(message, exc_info=exc_info)
            else:                    self.class_logger.debug(message)

    def _preprocess_X(self, X: Union[pd.DataFrame, np.ndarray], scale_data: bool = True) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=np.number)
            if X_numeric.shape[1] < X.shape[1]:
                self._log(f"Non-numeric columns detected. Using {X_numeric.shape[1]} numeric columns.", level="debug")
            if X_numeric.empty: raise ValueError("No numeric data found in DataFrame for clustering.")
            X_np = X_numeric.values.astype(np.float64)
        elif isinstance(X, np.ndarray):
            if not np.issubdtype(X.dtype, np.number): raise ValueError("Input NumPy array must be numeric.")
            X_np = X.astype(np.float64)
        else: raise TypeError("Input X must be a pandas DataFrame or a NumPy ndarray.")

        if X_np.size == 0:
            raise ValueError("Input data resulted in an empty array after numeric selection.")

        if not np.all(np.isfinite(X_np)):
            self._log("NaN/Inf values detected before scaling. Cleaning column-wise.", level="warning")
            for i in range(X_np.shape[1]):
                col_data = X_np[:, i]
                if np.any(~np.isfinite(col_data)):
                    finite_vals = col_data[np.isfinite(col_data)]
                    if len(finite_vals) > 0:
                        fill_value = np.median(finite_vals)
                        posinf_fill = np.max(finite_vals) if len(finite_vals) > 1 else fill_value
                        neginf_fill = np.min(finite_vals) if len(finite_vals) > 1 else fill_value
                        X_np[:, i] = np.nan_to_num(col_data, nan=fill_value, posinf=posinf_fill, neginf=neginf_fill)
                    else:
                        self._log(f"Column {i} is entirely non-finite. Filling with 0.0.", level="warning")
                        X_np[:, i] = 0.0

        if scale_data:
            self._log("Applying StandardScaler.", level="debug")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_np)
            if not np.all(np.isfinite(X_scaled)):
                self._log("Non-finite values detected AFTER scaling. Replacing with 0.0 (fallback).", level="warning")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            return X_scaled
        return X_np

    def run_kmeans(self, X_processed: np.ndarray, n_clusters: int, n_init: Union[str, int] = 'auto') -> Optional[np.ndarray]:
        if n_clusters <= 0: self._log(f"KMeans: n_clusters must be > 0, got {n_clusters}.", level="error"); return None
        if X_processed.shape[0] == 0: self._log("KMeans: empty data.", level="error"); return np.array([], dtype=int)
        actual_n_clusters = max(1, min(n_clusters, X_processed.shape[0]))
        if actual_n_clusters != n_clusters:
            self._log(f"KMeans: Adjusted n_clusters from {n_clusters} to {actual_n_clusters} (samples={X_processed.shape[0]}).", level="warning")
        self._log(f"Running KMeans with n_clusters={actual_n_clusters}.", level="debug")
        try:
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=self.random_state, n_init=n_init, init='k-means++')
            return kmeans.fit_predict(X_processed)
        except Exception as e:
            self._log(f"KMeans failed (k={actual_n_clusters}): {e}", level="error", exc_info=True)
            return None

    def run_minibatch_kmeans(self, X_processed: np.ndarray, n_clusters: int,
                             batch_size_ratio: float = 0.1, min_batch_size: int = 256,
                             max_batch_size: int = 4096, n_init: int = 3) -> Optional[np.ndarray]:
        if n_clusters <= 0: self._log(f"MiniBatchKMeans: n_clusters must be > 0, got {n_clusters}.", level="error"); return None
        if X_processed.shape[0] == 0: self._log("MiniBatchKMeans: empty data.", level="error"); return np.array([], dtype=int)
        actual_n_clusters = max(1, min(n_clusters, X_processed.shape[0]))
        if actual_n_clusters != n_clusters:
            self._log(f"MiniBatchKMeans: Adjusted n_clusters from {n_clusters} to {actual_n_clusters} (samples={X_processed.shape[0]}).", level="warning")
        n_samples = X_processed.shape[0]
        calculated_batch_size = int(n_samples * batch_size_ratio)
        effective_batch_size = min(max_batch_size, max(min_batch_size, calculated_batch_size))
        effective_batch_size = min(effective_batch_size, n_samples)
        if effective_batch_size <= 0 and n_samples > 0: effective_batch_size = n_samples
        self._log(f"Running MiniBatchKMeans with n_clusters={actual_n_clusters}, batch_size={effective_batch_size}.", level="debug")
        try:
            mbk = MiniBatchKMeans(
                n_clusters=actual_n_clusters, random_state=self.random_state,
                batch_size=effective_batch_size, n_init=n_init,
                max_iter=100,
            )
            return mbk.fit_predict(X_processed)
        except Exception as e:
            self._log(f"MiniBatchKMeans failed (k={actual_n_clusters}): {e}", level="error", exc_info=True)
            return None

    def run_agglomerative(self, X_processed: np.ndarray, n_clusters: int, linkage: str = 'ward') -> Optional[np.ndarray]:
        if n_clusters <= 0: self._log(f"Agglomerative: n_clusters must be > 0, got {n_clusters}.", level="error"); return None
        if X_processed.shape[0] == 0: self._log("Agglomerative: empty data.", level="error"); return np.array([], dtype=int)
        actual_n_clusters = max(1, min(n_clusters, X_processed.shape[0]))
        if actual_n_clusters != n_clusters:
            self._log(f"Agglomerative: Adjusted n_clusters from {n_clusters} to {actual_n_clusters} (samples={X_processed.shape[0]}).", level="warning")
        if linkage == 'ward' and X_processed.shape[0] < 2:
            self._log(f"Ward linkage requires >= 2 samples, got {X_processed.shape[0]}. Cannot run.", level="warning")
            return np.zeros(X_processed.shape[0], dtype=int) if actual_n_clusters == 1 else None
        self._log(f"Running Agglomerative Clustering: k={actual_n_clusters}, linkage='{linkage}'.", level="debug")
        try:
            metric = 'euclidean'
            agglo = AgglomerativeClustering(n_clusters=actual_n_clusters, metric=metric, linkage=linkage)
            return agglo.fit_predict(X_processed)
        except Exception as e:
            self._log(f"Agglomerative Clustering failed (k={actual_n_clusters}, linkage={linkage}): {e}", level="error", exc_info=True)
            return None

    def run_dbscan(self, X_processed: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Optional[np.ndarray]:
        if X_processed.shape[0] == 0: self._log("DBSCAN: empty data.", level="error"); return np.array([], dtype=int)
        self._log(f"Running DBSCAN: eps={eps}, min_samples={min_samples}.", level="debug")
        try:
            dbscan_n_jobs = self.n_jobs if self.n_jobs != 0 else None
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=dbscan_n_jobs)
            labels = dbscan.fit_predict(X_processed)
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            self._log(f"DBSCAN found {n_clusters_found} clusters, {n_noise} noise points ({n_noise*100.0/max(1,X_processed.shape[0]):.1f}%).", level="info")
            return labels
        except Exception as e:
            self._log(f"DBSCAN failed: {e}", level="error", exc_info=True)
            return None

    def run_hdbscan(self, X_processed: np.ndarray, min_cluster_size: int = 15, min_samples: Optional[int] = None,
                      allow_single_cluster: bool = False) -> Optional[np.ndarray]:
        if not self.use_hdbscan_if_available:
            self._log("HDBSCAN usage disabled by configuration.", level="info"); return None
        if not HDBSCAN_AVAILABLE or hdbscan is None:
            self._log("HDBSCAN library not available.", level="warning"); return None
        if X_processed.shape[0] == 0: self._log("HDBSCAN: empty data.", level="error"); return np.array([], dtype=int)
        actual_min_cluster_size = max(2, min_cluster_size)
        actual_min_samples = min_samples if min_samples is not None else max(1, int(actual_min_cluster_size * 0.5))
        actual_min_samples = max(1, min(actual_min_samples, actual_min_cluster_size))
        self._log(f"Running HDBSCAN: min_cluster_size={actual_min_cluster_size}, min_samples={actual_min_samples}.", level="debug")
        if X_processed.shape[0] < actual_min_cluster_size:
            self._log(f"HDBSCAN: samples ({X_processed.shape[0]}) < min_cluster_size ({actual_min_cluster_size}). Expect all noise.", level="warning")
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=actual_min_cluster_size, min_samples=actual_min_samples,
                allow_single_cluster=allow_single_cluster,
                core_dist_n_jobs=1
            )
            labels = clusterer.fit_predict(X_processed)
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            self._log(f"HDBSCAN found {n_clusters_found} clusters, {n_noise} noise points ({n_noise*100.0/max(1,X_processed.shape[0]):.1f}%).", level="info")
            return labels
        except Exception as e:
            self._log(f"HDBSCAN failed: {e}", level="error", exc_info=True)
            return np.full(X_processed.shape[0], -1, dtype=int) if X_processed.shape[0] > 0 else np.array([], dtype=int)

    def run_dp_gmm(self, X_processed: np.ndarray,
                   target_n_components: int = 15, # K* from find_optimal_k can be passed here
                   weight_concentration_prior: float = 0.1,
                   reg_covar: float = 1e-5,
                   mean_precision_prior: float = 0.01,
                   n_init_bgm: int = 1
                   ) -> Tuple[Optional[BayesianGaussianMixture], Dict[str, Any]]:
        """
        Fits a Bayesian Gaussian Mixture model (DP-GMM).
        target_n_components sets the upper bound for n_components in BGM.
        """
        self._log(f"Fitting DP-GMM (BayesianGaussianMixture, target_n_components={target_n_components}, reg_covar={reg_covar}).", level="info")
        n_samples, n_features = X_processed.shape
        if n_samples == 0:
            self._log("DP-GMM: empty data.", level="error")
            return None, {'error': 'Empty input data', 'effective_components': 0, 'labels': np.array([], dtype=int)}

        # Use target_n_components as the n_components for BGM initialization
        # BGM will attempt to fit up to this many components.
        actual_n_components_for_bgm = min(target_n_components, n_samples)
        actual_n_components_for_bgm = max(1, actual_n_components_for_bgm) # Must be >= 1
        self._log(f"DP-GMM effective n_components for BGM init: {actual_n_components_for_bgm}.", level="debug")

        try:
            bgm = BayesianGaussianMixture(
                n_components=actual_n_components_for_bgm, covariance_type='diag',
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=weight_concentration_prior,
                mean_precision_prior=mean_precision_prior,
                reg_covar=reg_covar,
                max_iter=300, n_init=n_init_bgm,
                random_state=self.random_state,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
                bgm.fit(X_processed)

            weights = bgm.weights_
            # Determine effective components based on weight threshold
            # Use actual_n_components_for_bgm for calculating threshold
            significant_threshold = 1.0 / (actual_n_components_for_bgm * 10.0)
            significant_threshold = min(0.01, max(1e-4, significant_threshold))
            effective_mask = weights > significant_threshold
            effective_n = int(np.sum(effective_mask))

            if effective_n == 0 and len(weights) > 0:
                self._log("DP-GMM: All components below threshold. Taking component with largest weight.", level="warning")
                effective_n = 1
                effective_mask = np.zeros_like(weights, dtype=bool)
                effective_mask[np.argmax(weights)] = True

            labels = bgm.predict(X_processed) if n_samples > 0 else np.array([], dtype=int)
            self.last_dp_gmm_model = bgm
            self.last_dp_gmm_labels = labels

            info = {
                'effective_components': effective_n, 'labels': labels,
                'model_converged': bgm.converged_, 'n_iter': bgm.n_iter_,
                'weights': weights[effective_mask].tolist() if effective_n > 0 else [],
                'means': bgm.means_[effective_mask].tolist() if effective_n > 0 else [],
                'covariances': bgm.covariances_[effective_mask].tolist() if effective_n > 0 else [],
                'bic_score': bgm.bic(X_processed) if hasattr(bgm, 'bic') else None,
                'aic_score': bgm.aic(X_processed) if hasattr(bgm, 'aic') else None,
                'lower_bound': bgm.lower_bound_
            }
            self._log(f"DP-GMM converged: {bgm.converged_} in {bgm.n_iter_} iters. Effective components found: {effective_n} (from target K={target_n_components}).", level="info")
            return bgm, info
        except Exception as e:
            self._log(f"Error during DP-GMM fitting: {e}", level="error", exc_info=True)
            return None, {'error': str(e), 'effective_components': 0, 'labels': np.zeros(n_samples, dtype=int) if n_samples > 0 else np.array([], dtype=int)}

    def find_optimal_k(self,
                       X_processed: np.ndarray,
                       k_range: Union[Sequence[int], range] = range(2, 11),
                       metric: str = 'silhouette',
                       base_clustering_algo: str = 'minibatch_kmeans',
                       sample_size_for_k_search: Optional[int] = None
                       ) -> int:
        actual_sample_size = sample_size_for_k_search if sample_size_for_k_search is not None else self.default_sample_size_for_k_search
        self._log(f"Finding optimal K using '{metric}' (Range: {min(k_range)}-{max(k_range)}, Algo: '{base_clustering_algo}', SampleSize: {actual_sample_size}).", level="info")
        n_samples_orig = X_processed.shape[0]

        X_eval = X_processed
        if n_samples_orig > actual_sample_size:
            self._log(f"Subsampling data from {n_samples_orig} to {actual_sample_size} for optimal K search.", level="debug")
            X_eval = sk_resample(X_processed, n_samples=actual_sample_size, random_state=self.random_state)

        n_eval_samples = X_eval.shape[0]
        valid_k_range = [k for k in k_range if 1 < k < n_eval_samples]

        if not valid_k_range:
            fallback_k = 2 if n_eval_samples >= 2 else 1
            self._log(f"k_range {list(k_range)} invalid or no K > 1 possible for sample size {n_eval_samples}. Defaulting K to {fallback_k}.", level="warning")
            return fallback_k

        scores = {}
        best_score = -np.inf if metric in ['silhouette', 'calinski_harabasz'] else np.inf
        optimal_k = min(valid_k_range)

        self._log(f"Evaluating K values: {valid_k_range}", level="debug")
        for k_val in valid_k_range:
            labels = None
            try:
                if base_clustering_algo == 'kmeans': labels = self.run_kmeans(X_eval, k_val)
                elif base_clustering_algo == 'minibatch_kmeans': labels = self.run_minibatch_kmeans(X_eval, k_val)
                else: self._log(f"Unsupported base_clustering_algo '{base_clustering_algo}'.", level="warning"); break

                if labels is None:
                    self._log(f"Base clustering '{base_clustering_algo}' failed for K={k_val}. Skipping.", level="warning")
                    continue

                unique_labels = np.unique(labels)
                n_unique_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

                if n_unique_clusters < 2:
                    self._log(f"Skipping metric evaluation for K={k_val}, resulted in {n_unique_clusters} valid cluster(s).", level="debug")
                    continue

                current_score: Optional[float] = None
                if metric == 'silhouette': current_score = silhouette_score(X_eval, labels)
                elif metric == 'davies_bouldin': current_score = davies_bouldin_score(X_eval, labels)
                elif metric == 'calinski_harabasz': current_score = calinski_harabasz_score(X_eval, labels)
                else: self._log(f"Unsupported metric '{metric}'. Stopping K search.", level="error"); break

                if current_score is not None and np.isfinite(current_score):
                    scores[k_val] = current_score
                    self._log(f"   K={k_val}, {metric.capitalize()} Score: {current_score:.4f}", level="debug")
                    higher_is_better = metric in ['silhouette', 'calinski_harabasz']
                    if higher_is_better and current_score > best_score:
                        best_score, optimal_k = current_score, k_val
                    elif not higher_is_better and current_score < best_score:
                        best_score, optimal_k = current_score, k_val
                else:
                    self._log(f"Metric calculation failed or resulted in non-finite score for K={k_val}.", level="warning")

            except Exception as e:
                self._log(f"Error evaluating K={k_val} with metric '{metric}': {e}", level="warning", exc_info=False)

        if not scores:
            self._log(f"Could not evaluate any K in range {valid_k_range}. Defaulting to {optimal_k}.", level="warning")
        else:
            self._log(f"Optimal K determined as {optimal_k} (Best Score for '{metric}': {best_score:.4f})", level="info")
        return optimal_k

    def run_ensemble_clustering(self,
                                X: Union[pd.DataFrame, np.ndarray],
                                n_clusters: int # Target K* determined externally
                               ) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, Any]]:
        self._log(f"Running Ensemble Clustering (Target K={n_clusters}).", level="info")
        X_processed = self._preprocess_X(X, scale_data=True)
        n_samples = X_processed.shape[0]

        if n_samples == 0:
            self._log("Ensemble: empty data.", level="error")
            return {}, {'error': 'Empty input data', 'labels': np.array([], dtype=int), 'n_clusters_found': 0}

        final_n_clusters = max(1, min(n_clusters, n_samples))
        if final_n_clusters != n_clusters:
            self._log(f"Adjusted ensemble K from {n_clusters} to {final_n_clusters} (samples: {n_samples}).", level="warning")

        if final_n_clusters == 1 and n_samples > 0:
            self._log("Ensemble target K=1. Assigning all points to cluster 0.", level="info")
            labels = np.zeros(n_samples, dtype=int)
            return {'single_cluster': labels}, {'final_labels': labels, 'n_clusters_found': 1, 'method': 'Single Cluster'}

        labels_dict: Dict[str, Optional[np.ndarray]] = {}
        labels_dict['kmeans'] = self.run_kmeans(X_processed, final_n_clusters)
        labels_dict['minibatch_kmeans'] = self.run_minibatch_kmeans(X_processed, final_n_clusters)

        if n_samples < self.ensemble_agglo_sample_threshold:
            if final_n_clusters >= 2:
                for linkage_type in ['ward', 'average', 'complete']:
                    if linkage_type == 'ward' and n_samples < 2: continue
                    labels_dict[f'agglo_{linkage_type}'] = self.run_agglomerative(X_processed, final_n_clusters, linkage=linkage_type)
        else:
            self._log(f"Skipping Agglomerative base clusterers (N={n_samples} >= threshold {self.ensemble_agglo_sample_threshold}).", level="info")

        if self.use_hdbscan_if_available:
            hdb_min_size = max(5, n_samples // (final_n_clusters * 5 if final_n_clusters > 1 else 20))
            hdb_min_size = max(2, min(hdb_min_size, n_samples // 2 if n_samples > 1 else 1))
            labels_dict['hdbscan'] = self.run_hdbscan(X_processed, min_cluster_size=hdb_min_size, allow_single_cluster=(final_n_clusters == 1))

        if self.last_dp_gmm_labels is not None and len(self.last_dp_gmm_labels) == n_samples:
            labels_dict['dp_gmm'] = self.last_dp_gmm_labels

        valid_partitions = {name: lbls for name, lbls in labels_dict.items()
                            if lbls is not None and len(lbls) == n_samples}

        if not valid_partitions:
            self._log("No successful base clusterings for ensemble. Cannot proceed.", level="error")
            return {}, {'error': 'All base clusterers failed', 'final_labels': np.zeros(n_samples, dtype=int), 'n_clusters_found': 1, 'method':'Fallback'}

        co_association_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        num_partitions_used = 0
        self._log(f"Building co-association matrix from {len(valid_partitions)} base partitions.", level="debug")
        for algo_name, labels in valid_partitions.items():
            num_partitions_used += 1
            for i in range(n_samples):
                if labels[i] != -1:
                    same_cluster_mask = (labels == labels[i]) & (labels != -1)
                    co_association_matrix[i, same_cluster_mask] += 1.0

        if num_partitions_used == 0:
            self._log("No non-trivial partitions found for co-association. Fallback needed.", level="warning")
            fallback_labels = labels_dict.get('kmeans', labels_dict.get('minibatch_kmeans'))
            if fallback_labels is None: fallback_labels = np.zeros(n_samples, dtype=int)
            return valid_partitions, {'error':'No valid partitions for consensus', 'final_labels': fallback_labels, 'n_clusters_found': len(np.unique(fallback_labels)), 'method': 'Fallback'}

        co_association_matrix /= num_partitions_used

        self._log(f"Performing consensus clustering (Target K={final_n_clusters}) from co-association matrix (N={n_samples}).", level="info")
        consensus_labels: Optional[np.ndarray] = None
        consensus_method: str = "N/A"

        try:
            if n_samples >= self.consensus_spectral_threshold and final_n_clusters >= 2:
                self._log(f"Using SpectralClustering for consensus (N={n_samples} >= threshold {self.consensus_spectral_threshold}).", level="info")
                affinity_matrix = np.nan_to_num(co_association_matrix, nan=0.0, posinf=1.0, neginf=0.0)
                affinity_matrix = np.maximum(affinity_matrix, 0)
                spectral_k = min(final_n_clusters, n_samples - 1 if n_samples > 1 else 1)
                if spectral_k < 2 and n_samples >=2 : # If n_samples = 1, spectral_k would be 1.
                     self._log(f"SpectralClustering needs K>=2, effective K is {spectral_k}. Using Agglomerative for consensus.", level="warning")
                     # Fallthrough to Agglomerative
                elif spectral_k >=2 : # Only if K>=2 proceed with spectral
                    consensus_spectral = SpectralClustering(
                        n_clusters=spectral_k, affinity='precomputed',
                        assign_labels='kmeans',
                        random_state=self.random_state, n_jobs=self.n_jobs
                    )
                    consensus_labels = consensus_spectral.fit_predict(affinity_matrix)
                    consensus_method = f"SpectralClustering (Target K={spectral_k})"
                # If spectral_k < 2, consensus_labels remains None, will try agglomerative

            # If Spectral was not used (either due to size or spectral_k < 2) or failed implicitly
            if consensus_labels is None:
                self._log(f"Using AgglomerativeClustering for consensus (N={n_samples}).", level="info")
                distance_matrix = np.maximum(1.0 - co_association_matrix, 0)
                np.fill_diagonal(distance_matrix, 0)

                # Ensure K for Agglomerative is valid
                agglo_k = final_n_clusters
                if agglo_k >= n_samples and n_samples > 0 : agglo_k = n_samples -1
                if agglo_k < 1 and n_samples > 0 : agglo_k = 1
                if agglo_k == 0 and n_samples == 0: agglo_k = 0 # No clusters for no samples
                elif agglo_k <= 0 : # If n_samples > 0 but agglo_k is still <=0
                     self._log(f"Agglomerative consensus K became invalid ({agglo_k}) for N={n_samples}. Defaulting to 1.", "warning")
                     agglo_k = 1


                if agglo_k > 0 or (agglo_k==0 and n_samples==0): # Proceed if K is valid
                    consensus_agglo = AgglomerativeClustering(
                        n_clusters=agglo_k, metric='precomputed', linkage='average'
                    )
                    consensus_labels = consensus_agglo.fit_predict(distance_matrix)
                    consensus_method = f"Agglomerative (Target K={agglo_k})"
                else: # Should not happen with above logic
                    raise ValueError("Agglomerative K calculation resulted in invalid K for consensus.")


            if consensus_labels is None: raise ValueError("Consensus clustering algorithm returned None.")

            actual_consensus_k = len(np.unique(consensus_labels)) - (1 if -1 in consensus_labels else 0)
            self._log(f"Consensus clustering ({consensus_method}) resulted in {actual_consensus_k} clusters.", level="info")
            sil_score, db_score, ch_score = None, None, None
            if actual_consensus_k > 1 and n_samples > actual_consensus_k:
                try:
                    sil_score = silhouette_score(X_processed, consensus_labels)
                    db_score = davies_bouldin_score(X_processed, consensus_labels)
                    ch_score = calinski_harabasz_score(X_processed, consensus_labels)
                    self._log(f"Consensus scores: Sil={sil_score:.3f}, DB={db_score:.3f}, CH={ch_score:.1f}", level="info")
                except ValueError as ve_metric:
                    self._log(f"Could not compute metrics for consensus labels: {ve_metric}", level="warning")

            consensus_info = {
                'final_labels': consensus_labels,
                'n_clusters_found': actual_consensus_k, 'method': consensus_method,
                'silhouette_score': sil_score, 'davies_bouldin_score': db_score,
                'calinski_harabasz_score': ch_score,
            }
            return valid_partitions, consensus_info

        except Exception as e_consensus:
            self._log(f"Consensus clustering failed: {e_consensus}", level="error", exc_info=True)
            fallback_labels = valid_partitions.get('kmeans', valid_partitions.get('minibatch_kmeans'))
            if fallback_labels is None: fallback_labels = np.zeros(n_samples, dtype=int)
            return valid_partitions, {'error': str(e_consensus), 'final_labels': fallback_labels, 'n_clusters_found': len(np.unique(fallback_labels)), 'method': 'Fallback'}

