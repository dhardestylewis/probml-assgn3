# -*- coding: utf-8 -*-
from probml.core.utils import get_logger, FAISS_AVAILABLE, SKDIM_AVAILABLE, PICARD_AVAILABLE
from probml.core.ica import PicardResult
from sklearn.exceptions import ConvergenceWarning

# File: dimensionality_reduction.py
import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional, Dict, Any, NamedTuple, List, Union
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.neighbors import NearestNeighbors


# --- Dependency Availability Check Placeholders ---
# Assume these might be defined in a previous cell. If not, define them here.
if 'FAISS_AVAILABLE' not in globals():
    FAISS_AVAILABLE = False
    try:
        import faiss
        FAISS_AVAILABLE = True
    except ImportError:
        pass

if 'SKDIM_AVAILABLE' not in globals():
    SKDIM_AVAILABLE = False
    try:
        from skdim.id import TwoNN # type: ignore
        SKDIM_AVAILABLE = True
    except ImportError:
        pass

if 'PICARD_AVAILABLE' not in globals():
    PICARD_AVAILABLE = False
    try:
        from picard import picard # type: ignore
        PICARD_AVAILABLE = True
    except ImportError:
        pass


logger = get_logger(__name__) # Module-level logger if needed outside class


class DimensionalityReducer:
    """
    Handles intrinsic dimension estimation and Independent Component Analysis (ICA).
    Uses n_jobs for compatible operations like k-NN.
    """
    def __init__(self,
                 random_state: Optional[int] = None,
                 verbose: bool = True,
                 n_jobs: int = -1, # For parallelizable operations (e.g., kNN)
                 use_advanced_methods: bool = True, # Use FAISS, skdim, Picard if available
                 prefer_gpu_for_faiss: bool = True, # Attempt GPU for FAISS kNN
                 max_ica_iter: int = 3000):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose
        self.n_jobs = n_jobs # Store n_jobs for use where applicable
        self.use_advanced_methods = use_advanced_methods
        self.prefer_gpu_for_faiss = prefer_gpu_for_faiss
        self.max_ica_iter = max_ica_iter
        self.logger = get_logger(self.__class__.__name__, verbose=self.verbose)

    def _log(self, message: str, level: str = "info", exc_info: bool = False):
        """Helper for conditional logging via the instance's logger."""
        # Renamed from logger to self.logger to avoid conflict with module logger
        if self.verbose:
            if level == "info":    self.logger.info(message)
            elif level == "warning": self.logger.warning(message)
            elif level == "error":   self.logger.error(message, exc_info=exc_info)
            else:                    self.logger.debug(message)

    def _preprocess_X(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Converts input to numpy array, cleans NaN/Inf, and scales."""
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=np.number)
            if X_numeric.shape[1] < X.shape[1]:
                self._log(f"Non-numeric columns found. Using only {X_numeric.shape[1]} numeric columns for DR.", level="warning")
            if X_numeric.empty: raise ValueError("No numeric data for DR.")
            X_np = X_numeric.values.astype(np.float64)
        elif isinstance(X, np.ndarray):
            if not np.issubdtype(X.dtype, np.number): raise ValueError("Input NumPy array must be numeric.")
            X_np = X.astype(np.float64)
        else: raise TypeError("Input X must be pandas DataFrame or NumPy array.")

        if not np.all(np.isfinite(X_np)):
            self._log("NaN/Inf values detected before DR scaling. Imputing with median.", level="warning")
            for j in range(X_np.shape[1]):
                col_data = X_np[:, j]
                if np.any(~np.isfinite(col_data)):
                    finite_vals = col_data[np.isfinite(col_data)]
                    fill_val = np.median(finite_vals) if len(finite_vals) > 0 else 0.0
                    # Use nan_to_num for comprehensive cleaning
                    X_np[:, j] = np.nan_to_num(col_data, nan=fill_val, posinf=fill_val, neginf=fill_val) # Fill infs also

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_np)

        if np.any(~np.isfinite(X_scaled)):
            self._log("NaNs/Infs present AFTER scaling in _preprocess_X. Likely constant columns. Imputing with 0.", level="error")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0) # Impute post-scaling as fallback

        return X_scaled

    def estimate_intrinsic_dimension_twonn(self, X_processed: np.ndarray) -> Optional[Dict[str, Any]]:
        """Estimates intrinsic dimension using TwoNN if available."""
        if SKDIM_AVAILABLE and self.use_advanced_methods:
            self._log("Attempting TwoNN for intrinsic dimension estimation.", level="info")
            if not np.all(np.isfinite(X_processed)):
                self._log("Input data for TwoNN contains NaN/Inf. Cannot proceed.", level="error")
                return None
            try:
                # Ensure TwoNN is imported if SKDIM_AVAILABLE is True
                from skdim.id import TwoNN # type: ignore
                estimator = TwoNN().fit(X_processed) # type: ignore
                dim_estimate = estimator.dimension_
                if not np.isfinite(dim_estimate) or dim_estimate <= 0:
                    self._log(f"TwoNN returned invalid dimension: {dim_estimate}. Fallback.", level="warning")
                    return None

                median_dim = int(round(max(1, dim_estimate)))
                # Use a simple heuristic for std/quantiles based on median
                std_dim = max(1.0, median_dim * 0.15) # Arbitrary std deviation guess
                q25 = max(1, int(round(median_dim - std_dim)))
                q75 = int(round(median_dim + std_dim))

                result = {
                    'median': median_dim, 'mean': float(dim_estimate), 'std': std_dim,
                    'q25': q25, 'q75': q75,
                    'raw_estimates': np.array([dim_estimate]), 'method': 'TwoNN'
                }
                self._log(f"TwoNN Intrinsic Dimension: Median={median_dim} (Mean={dim_estimate:.2f}, Est.Std={std_dim:.2f})", level="info")
                return result
            except Exception as e:
                self._log(f"TwoNN failed: {e}. Check input data (NaNs?) or library compatibility.", level="warning", exc_info=True)
        else:
            self._log("TwoNN not available or not used.", level="debug")
        return None

    def estimate_intrinsic_dimension_levina_bickel(self, X_processed: np.ndarray, k_neighbors: int = 20) -> Dict[str, Any]:
        """Estimates intrinsic dimension using Levina-Bickel MLE."""
        self._log(f"Using Levina-Bickel MLE (k_neighbors={k_neighbors}).", level="info")
        n_samples, n_features = X_processed.shape

        actual_k = min(k_neighbors, n_samples - 1)
        if actual_k <= 1:
            self._log(f"Not enough samples/k for Levina-Bickel (need k>1, have {actual_k}). Defaulting dim=1.", level="warning")
            return {'median': 1, 'mean': 1.0, 'std': 0.0, 'q25': 1, 'q75': 1, 'raw_estimates': np.array([1.0]), 'method': 'Levina-Bickel (Fallback)'}

        if not np.all(np.isfinite(X_processed)):
            self._log("Input data for Levina-Bickel kNN contains NaN/Inf. Cannot compute kNN.", level="error")
            return {'median': 1, 'mean': 1.0, 'std': 0.0, 'q25': 1, 'q75': 1, 'raw_estimates': np.array([1.0]), 'method': 'Levina-Bickel (Error)'}

        dists: Optional[np.ndarray] = None
        if FAISS_AVAILABLE and self.use_advanced_methods and n_samples > 1000 :
            self._log("Using FAISS for k-NN in Levina-Bickel.", level="info")
            try:
                import faiss # Import here again just in case
                X_32 = X_processed.astype('float32')
                d = X_32.shape[1]
                index_cpu = faiss.IndexFlatL2(d)
                index_to_use = index_cpu
                gpu_res = None

                if self.prefer_gpu_for_faiss and hasattr(faiss, 'StandardGpuResources'):
                    try:
                        gpu_res = faiss.StandardGpuResources()
                        index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)
                        index_to_use = index_gpu
                        self._log("FAISS using GPU for k-NN.", level="info")
                    except Exception as gpu_e:
                        self._log(f"FAISS GPU failed: {gpu_e}. Using CPU.", level="warning")
                        # index_to_use remains index_cpu

                index_to_use.add(X_32)
                dists, _ = index_to_use.search(X_32, actual_k + 1) # Include self

                # Clean up GPU resources explicitly if they were created
                if gpu_res is not None: del gpu_res; del index_to_use # Allow GC, index_gpu uses gpu_res

            except Exception as faiss_e:
                self._log(f"FAISS k-NN failed: {faiss_e}. Falling back to sklearn.", level="warning", exc_info=True)
                dists = None

        if dists is None:
            self._log(f"Using sklearn NearestNeighbors for k-NN (n_jobs={self.n_jobs}).", level="info")
            try:
                nn = NearestNeighbors(n_neighbors=actual_k + 1, n_jobs=self.n_jobs, algorithm='auto').fit(X_processed)
                dists, _ = nn.kneighbors(X_processed)
            except Exception as nn_e:
                self._log(f"NearestNeighbors failed: {nn_e}", level="error", exc_info=True)
                return {'median': 1, 'mean': 1.0, 'std': 0.0, 'q25': 1, 'q75': 1, 'raw_estimates': np.array([1.0]), 'method': 'Levina-Bickel (kNN Error)'}

        # Ensure distances are finite (should be if input was finite, but safety check)
        if not np.all(np.isfinite(dists)):
             self._log("Non-finite distances found AFTER kNN. Cleaning before Levina-Bickel calc.", level="warning")
             dists = np.nan_to_num(dists, nan=np.inf) # Treat NaN dist as Inf

        dists_k = dists[:, 1:actual_k+1] # Exclude self (0th neighbor)
        dists_k = np.maximum(dists_k, 1e-12) # Avoid log(0)
        T_k_i = dists_k[:, -1] # Distance to the actual_k th neighbor (k-th element, index k-1)

        # Check for zero distances to k-th neighbor which cause issues
        if np.any(T_k_i <= 1e-12):
             zero_dist_indices = np.where(T_k_i <= 1e-12)[0]
             self._log(f"Found {len(zero_dist_indices)} points with zero/near-zero distance to k-th neighbor ({actual_k}). This implies duplicate points or insufficient k. Estimates for these points may be unstable.", level="warning")

        log_ratios_sum = np.sum(np.log(T_k_i[:, np.newaxis] / dists_k[:, :-1]), axis=1) # Sum over j=1 to k-1

        with np.errstate(divide='ignore', invalid='ignore'):
            # Note: Levina-Bickel formula uses sum j=1 to k-1, so there are k-1 terms.
            # Formula: d = ( (1 / n_samples) * sum_i [ (k-1) / sum_{j=1}^{k-1} log( T_k(i) / T_j(i) ) ] )^-1
            # Or calculate per point: d_i = ( (k-1) / sum_log_ratios_i )
            dim_estimates_i = (actual_k - 1) / log_ratios_sum
            dim_estimates_i[~np.isfinite(dim_estimates_i)] = 1 # Handle inf/nan (e.g., if log_ratios_sum is 0)

        dim_estimates_i = np.maximum(1, dim_estimates_i)
        dim_estimates_i = np.minimum(dim_estimates_i, n_features) # Cap at original features

        raw_estimates = dim_estimates_i[np.isfinite(dim_estimates_i)]
        if len(raw_estimates) == 0:
             self._log("No finite dimension estimates from Levina-Bickel. Returning default.", level="error")
             raw_estimates = np.array([1.0])

        median_dim = int(round(max(1, np.median(raw_estimates))))
        mean_dim = float(np.mean(raw_estimates))
        std_dim = float(np.std(raw_estimates))
        q25, q75 = np.percentile(raw_estimates, [25, 75])
        q25 = int(round(max(1, q25))); q75 = int(round(max(1, q75)))

        result = {'median': median_dim, 'mean': mean_dim, 'std': std_dim, 'q25': q25, 'q75': q75,
                  'raw_estimates': raw_estimates, 'method': 'Levina-Bickel'}
        self._log(f"Levina-Bickel ID: Median={median_dim} (Mean={mean_dim:.2f}, Std={std_dim:.2f}) from {len(raw_estimates)} estimates.", level="info")
        return result

    def estimate_intrinsic_dimension(self, X: Union[pd.DataFrame, np.ndarray],
                                     methods: List[str] = ['twonn', 'levina'],
                                     k_neighbors_lb: int = 20) -> Dict[str, Any]:
        """Estimates intrinsic dimension using specified methods, preferring first successful."""
        self._log(f"\n--- Estimating Intrinsic Dimension (Methods: {methods}) ---", level="info")
        X_processed = self._preprocess_X(X) # Handles scaling and cleaning
        n_samples, n_features = X_processed.shape

        if n_samples < k_neighbors_lb + 1 or n_samples < 5 : # Check if enough samples for methods
            self._log(f"Low samples ({n_samples}). Defaulting dim=min(5, n_features).", level="warning")
            default_dim = max(1, min(5, n_features))
            return {'median': default_dim, 'mean': float(default_dim), 'std': 0.0, 'q25': default_dim, 'q75': default_dim,
                    'raw_estimates': np.array([float(default_dim)]), 'method': 'Fallback (Low Samples)'}

        result = None
        for method in methods:
            if method.lower() == 'twonn':
                result = self.estimate_intrinsic_dimension_twonn(X_processed)
            elif method.lower() == 'levina':
                result = self.estimate_intrinsic_dimension_levina_bickel(X_processed, k_neighbors=k_neighbors_lb)

            if result and result.get('median', 0) > 0:
                self._log(f"Using result from '{result.get('method', method)}' method.", level="info")
                return result

        self._log("All specified intrinsic dimension estimation methods failed. Defaulting.", level="error")
        default_dim = max(1, min(5, n_features))
        return {'median': default_dim, 'mean': float(default_dim), 'std': 0.0, 'q25': default_dim, 'q75': default_dim,
                'raw_estimates': np.array([float(default_dim)]), 'method': 'Fallback (All Methods Failed)'}

    def run_ica_picard(self, X_processed: np.ndarray, n_components: int) -> Optional[Tuple[np.ndarray, PicardResult]]:
        """Runs Picard ICA if available."""
        if PICARD_AVAILABLE and self.use_advanced_methods:
            self._log(f"Attempting Picard ICA with {n_components} components.", level="info")
            if n_components > X_processed.shape[1]: n_components = X_processed.shape[1]
            if n_components <=0: self._log("n_components <= 0 for Picard.", level="error"); return None
            try:
                # Import here again in case scope changed
                from picard import picard # type: ignore

                # Picard documentation usually expects (features, samples)
                # Let's try that first.
                X_T = X_processed.T
                K, W, S_transformed_T = picard(X_T, n_components=n_components, ortho=False, max_iter=self.max_ica_iter, tol=1e-5, random_state=self.random_state)

                # Picard returns sources S as (n_components, n_samples)
                # We want (n_samples, n_components) for consistency with sklearn
                S_transformed = S_transformed_T.T

                # Verify output shapes
                if S_transformed.shape != (X_processed.shape[0], n_components):
                    raise ValueError(f"Picard output S shape mismatch: expected {(X_processed.shape[0], n_components)}, got {S_transformed.shape}")
                # K is the unmixing matrix (components, features)
                if K.shape != (n_components, X_processed.shape[1]):
                    raise ValueError(f"Picard output K (unmixing) shape mismatch: expected {(n_components, X_processed.shape[1])}, got {K.shape}")
                # W is the mixing matrix (features, components)
                if W.shape != (X_processed.shape[1], n_components):
                    raise ValueError(f"Picard output W (mixing) shape mismatch: expected {(X_processed.shape[1], n_components)}, got {W.shape}")

                ica_result_obj = PicardResult(components_=K, mixing_=W, n_iter_=self.max_ica_iter) # Assuming max_iter is best guess for n_iter
                self._log(f"Picard ICA completed. Sources shape: {S_transformed.shape}", level="info")
                return S_transformed, ica_result_obj
            except Exception as e:
                self._log(f"Picard ICA failed: {e}. Trying FastICA.", level="warning", exc_info=True)
        else:
             self._log("Picard not available or not used.", level="debug")
        return None

    def run_ica_fastica(self, X_processed: np.ndarray, n_components: int) -> Optional[Tuple[np.ndarray, FastICA]]:
        """Runs FastICA, iterating max_iter budget if needed."""
        self._log(f"Using FastICA with {n_components} components.", level="info")
        if n_components > X_processed.shape[1]: n_components = X_processed.shape[1]
        if n_components <=0: self._log("n_components <= 0 for FastICA.", level="error"); return None

        iter_budgets = [500, 1500, self.max_ica_iter]
        ica_model = None; S_transformed = None

        for max_iter_budget in iter_budgets:
            self._log(f"Attempting FastICA with max_iter={max_iter_budget}...", level="debug")
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                ica_model_current = FastICA(
                    n_components=n_components, algorithm='parallel', whiten='unit-variance',
                    max_iter=max_iter_budget, tol=1e-4, random_state=self.random_state
                )
                try:
                    S_transformed = ica_model_current.fit_transform(X_processed)
                    ica_model = ica_model_current
                    if hasattr(ica_model, 'n_iter_') and ica_model.n_iter_ < max_iter_budget - 1 :
                        self._log(f"FastICA converged in {ica_model.n_iter_} iterations.", level="info")
                        break # Converged early
                    self._log(f"FastICA reached max_iter {max_iter_budget}.", level="info")
                    # Check if this is the last budget; if so, we accept this result even if it didn't converge early.
                    if max_iter_budget == iter_budgets[-1]:
                        self._log("FastICA completed using final max_iter budget.", level="info")
                        break
                except Exception as e:
                    self._log(f"FastICA failed (max_iter={max_iter_budget}): {e}", level="warning", exc_info=False) # Less verbose exc_info here
                    if max_iter_budget == iter_budgets[-1]:
                        self._log("FastICA failed on all attempts.", level="error")
                        return None # Failed entirely after trying all budgets
                    # continue to next budget if not the last one

        if S_transformed is not None and ica_model is not None:
            n_iter_final = getattr(ica_model, 'n_iter_', 'Unknown')
            self._log(f"FastICA completed. Sources shape: {S_transformed.shape}, Iterations: {n_iter_final}", level="info")
            return S_transformed, ica_model

        self._log("FastICA did not produce a result.", level="error")
        return None

    def run_ica(self, X: Union[pd.DataFrame, np.ndarray], n_components: int) -> Optional[Tuple[np.ndarray, Any]]:
        """Runs ICA, trying Picard then FastICA."""
        self._log(f"\n--- Running ICA for {n_components} Components ---", level="info")
        X_processed = self._preprocess_X(X) # Handles scaling and cleaning

        if n_components <= 0 : self._log("n_components must be > 0 for ICA.", level="error"); return None
        if X_processed.shape[0] < n_components : self._log(f"Samples ({X_processed.shape[0]}) < n_components ({n_components}). ICA may fail or produce unreliable results.", level="warning")
        if X_processed.shape[1] < n_components :
            self._log(f"Features ({X_processed.shape[1]}) < n_components ({n_components}). Reducing n_components to {X_processed.shape[1]}.", level="warning")
            n_components = X_processed.shape[1]
        if n_components <= 0: self._log("Cannot run ICA with <= 0 components after adjustment.", level="error"); return None

        # Try Picard first if available and preferred
        picard_output = None
        if PICARD_AVAILABLE and self.use_advanced_methods:
             picard_output = self.run_ica_picard(X_processed, n_components)
        if picard_output:
             self._log("ICA successful using Picard.", level="info")
             return picard_output

        # Fallback to FastICA
        self._log("Picard failed or not used. Falling back to FastICA.", level="info")
        fastica_output = self.run_ica_fastica(X_processed, n_components)
        if fastica_output:
             self._log("ICA successful using FastICA.", level="info")
             return fastica_output

        self._log("ICA failed with all available methods.", level="error")
        return None

    def interpret_ica_components(self, ika_model: Any, feature_names: List[str], top_n: int = 5):
        """Interprets ICA components by showing top contributing features."""
        self._log("\n--- Interpreting ICA Components ---", level="info")

        components_matrix = None
        if isinstance(ika_model, FastICA) and hasattr(ika_model, 'components_'):
            components_matrix = ika_model.components_ # Unmixing matrix (n_components, n_features)
            model_type = "FastICA"
        elif isinstance(ika_model, PicardResult) and hasattr(ika_model, 'components_'):
             components_matrix = ika_model.components_ # Unmixing matrix K (n_components, n_features)
             model_type = "Picard"
        else:
             self._log("ICA model type not recognized or missing 'components_'. Cannot interpret.", level="warning")
             return

        if components_matrix is None:
            self._log("Could not retrieve components matrix from ICA model.", level="error")
            return

        n_ica_components, n_features_in_ica = components_matrix.shape

        if len(feature_names) != n_features_in_ica:
            self._log(f"Mismatch: provided feature names ({len(feature_names)}) vs ICA model features ({n_features_in_ica}). Cannot interpret accurately.", level="error")
            # Optionally try to proceed with limited names, or return
            # feature_names = [f"Feature_{i}" for i in range(n_features_in_ica)] # Example fallback
            return

        self._log(f"Interpreting {n_ica_components} components from {model_type} (Shape: {components_matrix.shape})", level="info")
        for i in range(n_ica_components):
            comp_vector = components_matrix[i, :]
            abs_weights = np.abs(comp_vector)
            # Use argsort for indices, then slice for top_n
            sorted_indices = np.argsort(abs_weights)[::-1][:min(top_n, n_features_in_ica)]

            interpretation = []
            for idx in sorted_indices:
                weight = comp_vector[idx]
                sign = '+' if weight >= 0 else '-'
                # Format nicely
                interpretation.append(f"{sign}{feature_names[idx]} ({abs(weight):.3f})")

            self._log(f"   Component {i+1}: {', '.join(interpretation)}")

