# -*- coding: utf-8 -*-
"""
Independent Component Analysis (ICA) module.

Provides a unified interface for ICA using Picard (if available) 
or scikit-learn's FastICA as fallback.
"""
import numpy as np
import logging
from typing import NamedTuple, Optional, Dict, Any, List

# Try to use Picard ICA for performance; fallback to scikit-learn FastICA
try:
    from picard import picard
    PICARD_AVAILABLE = True
except ImportError:
    PICARD_AVAILABLE = False
    from sklearn.decomposition import FastICA

logger = logging.getLogger(__name__)


class PicardResult(NamedTuple):
    """
    Stores results from Picard ICA.
    
    Attributes:
        components_: The unmixing matrix (n_components, n_features)
        mixing_: The mixing matrix (n_features, n_components)
        n_iter_: Number of iterations to convergence
    """
    components_: np.ndarray
    mixing_: np.ndarray
    n_iter_: int


class ICAModule:
    """
    Wrapper for Independent Component Analysis (ICA).
    
    Uses Picard if available, otherwise falls back to scikit-learn's FastICA.
    
    Args:
        random_state: Random seed for reproducibility
        verbose: Whether to print progress messages
    """
    
    def __init__(self, random_state: Optional[int] = None, verbose: bool = False):
        self.random_state = random_state
        self.verbose = verbose
        if PICARD_AVAILABLE:
            logger.info("ICAModule: Using Picard ICA implementation.")
        else:
            logger.info("ICAModule: Picard not available, falling back to scikit-learn FastICA.")

    def run_ica(
        self,
        X: np.ndarray,
        n_components: int,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform ICA on the input data X.

        Args:
            X: Input data matrix of shape (n_samples, n_features)
            n_components: Number of independent components to estimate
            feature_names: Names of the original features (for metadata)

        Returns:
            Dictionary containing:
                - 'ica_sources': np.ndarray of shape (n_samples, n_components)
                - 'ica_mixing_matrix': np.ndarray of shape (n_features, n_components)
                - 'ica_model': underlying model or info dict
        """
        n_samples, n_features = X.shape
        if n_samples < 2 or n_features < 1:
            logger.warning("ICAModule.run_ica: Insufficient data for ICA. Returning empty sources.")
            return {
                'ica_sources': np.empty((n_samples, 0)),
                'ica_mixing_matrix': np.empty((n_features, 0)),
                'ica_model': None
            }

        # Center the data
        X_centered = X - np.mean(X, axis=0, keepdims=True)

        if PICARD_AVAILABLE:
            # Picard expects shape (n_features, n_samples)
            try:
                W, S, info = picard(
                    Y=X_centered.T,
                    ortho=False,
                    orth_wts=False,
                    fun="logcosh",
                    max_iter=200,
                    tol=1e-6,
                    random_state=self.random_state,
                    verbose=self.verbose
                )
                # S has shape (n_components, n_samples)
                sources = S.T[:, :n_components]
                mixing = np.linalg.pinv(W).T[:, :n_components]
                model = {'unmixing_matrix': W, 'info': info}
            except Exception as e:
                logger.error(f"ICAModule.run_ica: Picard ICA failed: {e}", exc_info=True)
                return {
                    'ica_sources': np.empty((n_samples, 0)),
                    'ica_mixing_matrix': np.empty((n_features, 0)),
                    'ica_model': None
                }
        else:
            # FastICA fallback
            try:
                ica = FastICA(
                    n_components=n_components,
                    random_state=self.random_state,
                    max_iter=200,
                    tol=1e-6
                )
                sources = ica.fit_transform(X_centered)
                mixing = ica.mixing_
                model = ica
            except Exception as e:
                logger.error(f"ICAModule.run_ica: FastICA failed: {e}", exc_info=True)
                return {
                    'ica_sources': np.empty((n_samples, 0)),
                    'ica_mixing_matrix': np.empty((n_features, 0)),
                    'ica_model': None
                }

        result = {
            'ica_sources': sources,
            'ica_mixing_matrix': mixing,
            'ica_model': model
        }
        if feature_names is not None:
            result['feature_names'] = feature_names[:n_features]

        logger.info(f"ICAModule: Extracted {sources.shape[1]} independent components.")
        return result


__all__ = ["PicardResult", "ICAModule", "PICARD_AVAILABLE"]
