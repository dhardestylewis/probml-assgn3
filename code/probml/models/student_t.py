# -*- coding: utf-8 -*-
"""
Student-T Mixture Model for heavy-tailed prior distributions.
"""
from typing import Tuple

from probml.core.utils import get_logger
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, digamma # For direct use of log gamma and digamma
from scipy.stats import t as scipy_t # For comparison or alternative PDF calculation
import warnings

# Assuming utils.py is in the same directory or accessible via PYTHONPATH
# from .utils import get_logger

logger = get_logger(__name__)

STUDENT_T_AVAILABLE = True

class StudentTMixture:
    """
    Mixture of Student's t-distributions fitted using the EM algorithm.
    Assumes diagonal covariance matrices (independent features for each component).
    """

    def __init__(self, n_components=3, max_iter=150, tol=1e-4,
                 min_df=2.0, max_df=150.0, random_state=None, reg_covar=1e-6):
        if n_components < 1:
            raise ValueError("Number of components must be at least 1.")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.min_df = min_df # Minimum degrees of freedom (Fix for Issue 2)
        self.max_df = max_df # Maximum degrees of freedom
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.reg_covar = reg_covar # Regularization for scale/covariance (Fix for Issue 2)

        self.weights_ = None
        self.means_ = None
        self.scales_ = None # Standard deviations (sqrt of variances)
        self.dfs_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = -np.inf

    def _init_params(self, X: np.ndarray):
        """Initialize parameters using k-means++ for means."""
        n_samples, n_features = X.shape
        if n_samples < self.n_components:
            # Fallback: reduce n_components if not enough samples
            logger.warning(f"Number of samples ({n_samples}) is less than n_components ({self.n_components}). "
                           f"Reducing n_components to {n_samples}.")
            self.n_components = n_samples
            if self.n_components == 0: # Should not happen if X is not empty
                 raise ValueError("Cannot initialize parameters with 0 components from 0 samples.")


        # K-means++ initialization for means
        centers = np.zeros((self.n_components, n_features))
        if n_samples > 0:
            centers[0] = X[self.rng.choice(n_samples)]
            distances = np.full(n_samples, np.inf)

            for i in range(1, self.n_components):
                dist_sq = np.sum((X - centers[i-1])**2, axis=1)
                distances = np.minimum(distances, dist_sq)

                # Handle cases where all distances are zero (e.g. all points are identical)
                sum_distances = np.sum(distances)
                if sum_distances == 0 : # if all points are the same, or remaining points are same
                    probs = np.ones(n_samples) / n_samples
                else:
                    probs = distances / sum_distances

                cumulative_probs = np.cumsum(probs)
                # Ensure probabilities sum to 1, handle potential floating point issues
                if cumulative_probs[-1] == 0 : # if all probs are 0, this can happen if distances are all 0
                    center_idx = self.rng.choice(n_samples) # pick randomly
                else:
                    cumulative_probs /= cumulative_probs[-1]
                    r = self.rng.random()
                    center_idx = np.searchsorted(cumulative_probs, r, side='right')

                center_idx = min(center_idx, n_samples - 1) # Ensure index is within bounds
                centers[i] = X[center_idx]
        else: # Should not be reached if n_samples < self.n_components raised error correctly
            # Fallback if n_samples is 0 somehow
            centers = np.zeros((self.n_components, n_features))


        self.means_ = centers
        self.weights_ = np.ones(self.n_components) / self.n_components

        # Initialize scales (std devs)
        # Use overall std dev for initial scales, add reg_covar for stability
        # Ensure std_X is not zero
        std_X = np.std(X, axis=0)
        if np.any(std_X == 0): # If any feature has zero variance
            logger.warning("One or more features have zero variance. Initializing scales with small default.")
            # Create scales with a small default value where std_X is 0
            feature_scales = np.where(std_X == 0, np.sqrt(self.reg_covar), std_X)
        else:
            feature_scales = std_X

        self.scales_ = np.ones((self.n_components, n_features)) * np.maximum(feature_scales, np.sqrt(self.reg_covar))

        # Initialize DoF (Fix for Issue 2 - ensure reasonable start)
        self.dfs_ = np.ones(self.n_components) * np.clip(5.0, self.min_df, self.max_df)


    def _log_t_pdf_component(self, X: np.ndarray, mean_k: np.ndarray, scale_k: np.ndarray, df_k: float) -> np.ndarray:
        """
        Calculate log PDF of a single multivariate Student's t-distribution component
        (assuming diagonal covariance).
        X: shape (n_samples, n_features)
        mean_k: shape (n_features,)
        scale_k: shape (n_features,) - standard deviations
        df_k: scalar
        Returns: log_pdf_val of shape (n_samples,)
        """
        n_samples, n_features = X.shape

        # Ensure parameters are valid and add regularization to scale
        df_safe = np.maximum(df_k, self.min_df)
        scale_safe = np.maximum(scale_k, np.sqrt(self.reg_covar)) # scale is std_dev

        # Log PDF using scipy.stats.t for simplicity and robustness per dimension, then sum
        # This assumes diagonal covariance (features are independent within a component)
        log_pdf_features = np.zeros_like(X)
        for j in range(n_features):
            log_pdf_features[:, j] = scipy_t.logpdf(X[:, j], df=df_safe, loc=mean_k[j], scale=scale_safe[j])

        return np.sum(log_pdf_features, axis=1)

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Expectation step: Calculate responsibilities and log-likelihood."""
        n_samples = X.shape[0]
        log_prob_norm = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            # Add small epsilon to weights to prevent log(0)
            weight_k = np.maximum(self.weights_[k], 1e-9)
            log_prob_norm[:, k] = np.log(weight_k) + \
                                  self._log_t_pdf_component(X, self.means_[k], self.scales_[k], self.dfs_[k])

        # Log-sum-exp trick for numerical stability
        log_likelihood_per_sample = np.logaddexp.reduce(log_prob_norm, axis=1) # More stable

        with np.errstate(divide='ignore', invalid='ignore'): # Handle potential division by zero if all log_prob_norm are -inf
            log_responsibilities = log_prob_norm - log_likelihood_per_sample[:, np.newaxis]
            responsibilities = np.exp(log_responsibilities)

        # Normalize responsibilities to sum to 1, handle cases where sum is 0 (all components had 0 prob)
        resp_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities = np.divide(responsibilities, resp_sum, out=np.zeros_like(responsibilities), where=resp_sum!=0)

        # If any sample has all-zero responsibilities (e.g., numerical underflow for all components),
        # assign it equally among components to prevent NaN issues in M-step.
        all_zero_resp_mask = (resp_sum == 0).flatten()
        if np.any(all_zero_resp_mask):
            logger.warning(f"{np.sum(all_zero_resp_mask)} samples had zero responsibility for all components. Re-distributing.")
            responsibilities[all_zero_resp_mask, :] = 1.0 / self.n_components


        total_log_likelihood = np.sum(log_likelihood_per_sample[np.isfinite(log_likelihood_per_sample)])
        if not np.isfinite(total_log_likelihood): # Fallback if sum is not finite
            total_log_likelihood = -np.inf if self.log_likelihood_ == -np.inf else self.log_likelihood_


        return responsibilities, total_log_likelihood

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """Maximization step: Update model parameters."""
        n_samples, n_features = X.shape

        # Add epsilon to prevent division by zero
        nk = np.sum(responsibilities, axis=0) + 1e-9 # Sum of responsibilities for each component

        # Update weights
        self.weights_ = nk / n_samples
        self.weights_ = np.maximum(self.weights_, 1e-9) # Ensure weights are not zero
        self.weights_ /= np.sum(self.weights_) # Normalize

        # Update means
        self.means_ = np.dot(responsibilities.T, X) / nk[:, np.newaxis]

        # Update scales (std devs) and dfs
        for k in range(self.n_components):
            if nk[k] < 1e-6 : # Skip update for collapsed components
                logger.warning(f"Component {k} has negligible responsibility ({nk[k]:.2e}). Skipping its M-step update.")
                self.scales_[k] = np.maximum(self.scales_[k], np.sqrt(self.reg_covar)) # Keep old scale but ensure regularized
                self.dfs_[k] = np.clip(self.dfs_[k], self.min_df, self.max_df) # Keep old df but ensure bounded
                continue

            diff = X - self.means_[k] # (n_samples, n_features)

            # Calculate u_ik weights for M-step update (Peel & McLachlan, 2000)
            # u_ik = (v_k + d) / (v_k + delta_ik^2)
            # where delta_ik^2 is Mahalanobis distance. For diagonal cov:
            # delta_ik_j^2 = (x_ij - mu_kj)^2 / sigma_kj^2
            # sum_j (delta_ik_j^2)

            # Ensure scale_k is not zero and includes regularization
            current_scale_k = np.maximum(self.scales_[k], np.sqrt(self.reg_covar))
            mahal_sq_terms = (diff / current_scale_k)**2 # (n_samples, n_features)
            mahal_sq_sum = np.sum(mahal_sq_terms, axis=1) # (n_samples,)

            current_df_k = np.clip(self.dfs_[k], self.min_df, self.max_df)
            u_ik = (current_df_k + n_features) / (current_df_k + mahal_sq_sum + 1e-9) # (n_samples,)

            # Check for NaNs or Infs in u_ik
            if np.any(~np.isfinite(u_ik)):
                logger.warning(f"Warning: Non-finite u_ik encountered for component {k}. Using mean of valid u_ik or 1.0.")
                u_ik[~np.isfinite(u_ik)] = np.nanmean(u_ik[np.isfinite(u_ik)]) if np.any(np.isfinite(u_ik)) else 1.0


            # Update scales (variances first, then sqrt)
            weighted_u_ik_resp = responsibilities[:, k] * u_ik # (n_samples,)
            sum_weighted_u_ik_resp = np.sum(weighted_u_ik_resp) + 1e-9

            if sum_weighted_u_ik_resp <= 1e-9: # If sum is effectively zero
                 logger.warning(f"Sum of weighted_u_ik_resp is non-positive for component {k}. Scale update might be unstable.")
                 # Keep previous scale or reset to a regularized version of global std
                 self.scales_[k] = np.maximum(self.scales_[k], np.sqrt(self.reg_covar))

            else:
                new_variance_k = np.sum(weighted_u_ik_resp[:, np.newaxis] * (diff ** 2), axis=0) / sum_weighted_u_ik_resp
                # Apply regularization to variance before sqrt
                self.scales_[k] = np.sqrt(np.maximum(new_variance_k, self.reg_covar))


            # Update degrees of freedom using optimization (Equation 20, Peel & McLachlan 2000)
            # Requires solving: -digamma(nu/2) + log(nu/2) + 1 - E_q[log u_ik] + E_q[u_ik] + digamma((nu+d)/2) - log((nu+d)/2) = 0
            # where E_q[log u_ik] = sum(r_ik * log u_ik) / sum(r_ik)
            # and   E_q[u_ik] = sum(r_ik * u_ik) / sum(r_ik)

            resp_k = responsibilities[:, k] # (n_samples,)
            # valid_idx = resp_k > 1e-9 # Use samples with some responsibility for this component
            # if not np.any(valid_idx):
            #     logger.warning(f"No samples with significant responsibility for component {k}. Skipping df update.")
            #     self.dfs_[k] = np.clip(self.dfs_[k], self.min_df, self.max_df)
            #     continue

            # resp_k_valid = resp_k[valid_idx]
            # u_ik_valid = u_ik[valid_idx]
            # log_u_ik_valid = np.log(np.maximum(u_ik_valid, 1e-9)) # Avoid log(0)

            # eq_log_u_ik = np.sum(resp_k_valid * log_u_ik_valid) / np.sum(resp_k_valid)
            # eq_u_ik = np.sum(resp_k_valid * u_ik_valid) / np.sum(resp_k_valid)

            # Simplified from scikit-learn's GaussianMixture for t-distribution M-step for df
            # (often df is fixed or updated with simpler heuristics due to complexity)
            # Here, we use the optimization approach as in the original code but with safeguards.

            # Define optimization function for degrees of freedom
            def neg_df_objective(nu_scalar):
                if nu_scalar <= self.min_df / 2: # Search in log space or transform to ensure positivity
                    return np.inf

                # Calculate E_q[log u_ik] and E_q[u_ik] for current nu_scalar via u_ik definition
                # This is tricky because u_ik itself depends on nu.
                # The original paper's objective function is simpler if E_q terms are treated as constants from E-step
                # Let's use the structure from the provided code which is common.

                # Recalculate u_ik terms based on nu_scalar for the objective function
                # This is what makes the optimization tricky.
                # We need E_rik_uik and E_rik_log_uik where uik depends on nu
                # For this component k:
                mahal_sq_sum_k = np.sum(( (X - self.means_[k]) / np.maximum(self.scales_[k], np.sqrt(self.reg_covar)) )**2, axis=1)
                u_ik_for_nu_opt = (nu_scalar + n_features) / (nu_scalar + mahal_sq_sum_k + 1e-9)

                # Ensure u_ik_for_nu_opt is positive for log
                log_u_ik_for_nu_opt = np.log(np.maximum(u_ik_for_nu_opt, 1e-9))

                # Weighted averages by responsibilities
                avg_log_u_ik = np.sum(responsibilities[:, k] * log_u_ik_for_nu_opt) / nk[k]
                avg_u_ik = np.sum(responsibilities[:, k] * u_ik_for_nu_opt) / nk[k]


                # Objective function based on Q2 w.r.t nu_k (Peel and McLachlan, 2000, eq. between (19) and (20))
                # Maximize: -digamma(nu/2) + log(nu/2) - avg_log_u_ik + avg_u_ik -1 + digamma((nu+d)/2) - log((nu+d)/2)
                # (Note: the constant +1 in the original text's neg_q2_df seems to be absorbed)

                val = (
                    -digamma(nu_scalar / 2.0)
                    + np.log(nu_scalar / 2.0)
                    - avg_log_u_ik
                    + avg_u_ik
                    - 1.0 # This -1.0 helps align with typical forms.
                    + digamma((nu_scalar + n_features) / 2.0)
                    - np.log((nu_scalar + n_features) / 2.0)
                )
                return -val # We minimize the negative of the objective

            initial_df_k = np.clip(self.dfs_[k], self.min_df, self.max_df)
            try:
                # Bounds for df, ensuring df > 0 (or min_df) and not excessively large
                opt_result = minimize(neg_df_objective, initial_df_k, method='L-BFGS-B',
                                      bounds=[(self.min_df, self.max_df)])
                if opt_result.success and np.isfinite(opt_result.x[0]):
                    self.dfs_[k] = np.clip(opt_result.x[0], self.min_df, self.max_df)
                else:
                    logger.warning(f"DF optimization failed for component {k}. Status: {opt_result.message}. Keeping previous df {initial_df_k:.2f}.")
                    self.dfs_[k] = initial_df_k # Keep old value if optimization fails
            except Exception as e:
                logger.error(f"Error in DF optimization for component {k}: {e}. Keeping previous df {initial_df_k:.2f}.")
                self.dfs_[k] = initial_df_k

        # Final parameter checks after M-step
        self.scales_ = np.maximum(self.scales_, np.sqrt(self.reg_covar)) # Ensure scales are positive and regularized
        self.dfs_ = np.clip(self.dfs_, self.min_df, self.max_df) # Ensure DoF are within bounds

    def fit(self, X: np.ndarray):
        """Fit the mixture model to the data X."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] == 0:
            logger.warning("Fitting on empty data. Model will not be fitted.")
            self.converged_ = False # Or True, depending on desired state for an empty fit
            # Set attributes to indicate no fit or default values
            self.weights_ = np.ones(self.n_components) / self.n_components if self.n_components > 0 else np.array([])
            self.means_ = np.zeros((self.n_components, X.shape[1])) if X.shape[1] > 0 and self.n_components > 0 else np.array([[]])
            self.scales_ = np.ones((self.n_components, X.shape[1])) if X.shape[1] > 0 and self.n_components > 0 else np.array([[]])
            self.dfs_ = np.ones(self.n_components) * self.min_df if self.n_components > 0 else np.array([])
            return self

        # Robustness: Check for NaNs or Infs in input X (Issue 1 related)
        if not np.all(np.isfinite(X)):
            logger.error("Input data X contains NaN or Inf values. Cannot fit StudentTMixture.")
            # Optionally, clean X here or raise an error
            # X = X[np.all(np.isfinite(X), axis=1)]
            # if X.shape[0] == 0:
            #     logger.error("All rows removed due to NaN/Inf. Cannot fit.")
            #     return self
            raise ValueError("Input data X for StudentTMixture contains NaN or Inf values.")


        try:
            self._init_params(X)
        except ValueError as e:
            logger.error(f"Error during initialization: {e}. Model cannot be fitted.")
            self.converged_ = False
            return self

        prev_ll = -np.inf
        self.converged_ = False

        for iteration in range(self.max_iter):
            self.n_iter_ = iteration + 1
            try:
                responsibilities, current_ll = self._e_step(X)

                if not np.isfinite(current_ll) or np.any(~np.isfinite(responsibilities)):
                    logger.warning(f"Warning: Non-finite values in E-step (Iter {self.n_iter_}, LL: {current_ll}). Stopping.")
                    # Attempt to recover or break
                    if not np.isfinite(current_ll) and prev_ll != -np.inf :
                        current_ll = prev_ll # Keep last good log-likelihood
                    else: # Cannot recover, break
                        break


                self._m_step(X, responsibilities)

                # Parameter sanity check after M-step
                if (np.any(~np.isfinite(self.weights_)) or
                    np.any(~np.isfinite(self.means_)) or
                    np.any(~np.isfinite(self.scales_)) or
                    np.any(~np.isfinite(self.dfs_))):
                    logger.warning(f"Warning: Non-finite parameters after M-step (Iter {self.n_iter_}). Stopping.")
                    break # Stop if parameters become non-finite

                delta_ll = current_ll - prev_ll
                # Check convergence, allow for slight decreases if tol is relative
                # and prev_ll is very small.
                abs_delta_ll = abs(delta_ll)

                converged_abs = abs_delta_ll < self.tol
                converged_rel = False
                if np.isfinite(prev_ll) and abs(prev_ll) > 1e-6 and np.isfinite(delta_ll): # prev_ll not zero or -inf
                    converged_rel = abs(delta_ll / prev_ll) < self.tol

                # If LL decreases significantly, it might indicate instability
                if delta_ll < -self.tol * 10 and iteration > 5 : # Allow some initial fluctuation
                    logger.warning(f"Log-likelihood decreased significantly from {prev_ll:.2f} to {current_ll:.2f} at iter {self.n_iter_}. Stopping.")
                    break


                if converged_abs or converged_rel:
                    self.converged_ = True
                    logger.info(f"Converged in {self.n_iter_} iterations. Log-likelihood: {current_ll:.4f}")
                    break

                prev_ll = current_ll
                self.log_likelihood_ = current_ll

            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
                logger.error(f"Numerical error during EM iteration {self.n_iter_}: {e}. Stopping.")
                # Consider re-initializing or breaking more gracefully
                break
            except Exception as e: # Catch any other unexpected error
                logger.error(f"Unexpected error in EM iteration {self.n_iter_}: {e}. Stopping.")
                break


        if not self.converged_ and self.n_iter_ == self.max_iter:
            logger.warning(f"EM algorithm did not converge within {self.max_iter} iterations. Final LL: {self.log_likelihood_:.4f}")
        elif not self.converged_ :
             logger.warning(f"EM algorithm stopped prematurely after {self.n_iter_} iterations due to issues. Final LL: {self.log_likelihood_:.4f}")


        # Ensure final parameters are valid, even if not converged
        if self.weights_ is not None:
            self.weights_ = np.maximum(self.weights_, 1e-9)
            self.weights_ /= np.sum(self.weights_)
        if self.scales_ is not None:
            self.scales_ = np.maximum(self.scales_, np.sqrt(self.reg_covar))
        if self.dfs_ is not None:
            self.dfs_ = np.clip(self.dfs_, self.min_df, self.max_df)

        # If any component weight is near zero, its parameters might be unreliable
        if self.weights_ is not None and np.any(self.weights_ < 1e-5):
            logger.info("Some components have very small weights. Their parameters might be less reliable.")

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculate the log probability density for each sample under the mixture."""
        if self.weights_ is None or self.means_ is None or self.scales_ is None or self.dfs_ is None:
            raise ValueError("Model has not been fitted yet or fitting failed.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] == 0:
            return np.array([])

        # Ensure input X is finite
        if not np.all(np.isfinite(X)):
            logger.warning("Input X for score_samples contains NaN/Inf. Results may be unpredictable.")
            # Decide on a strategy: error, clean, or proceed with caution.
            # For now, proceed, but PDF might be NaN/Inf for those rows.

        log_prob_norm = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            weight_k = np.maximum(self.weights_[k], 1e-9) # Smallest possible weight
            log_pdf_k = self._log_t_pdf_component(X, self.means_[k], self.scales_[k], self.dfs_[k])

            # Handle non-finite log_pdf_k (e.g., from extreme values in X)
            log_pdf_k[~np.isfinite(log_pdf_k)] = -np.inf # Treat non-finite pdf as zero probability
            log_prob_norm[:, k] = np.log(weight_k) + log_pdf_k

        # Log-sum-exp for stability
        max_log_prob = np.max(log_prob_norm, axis=1, keepdims=True)
        # Handle cases where all log_prob_norm are -inf for a sample
        finite_max_log_prob = np.where(np.isinf(max_log_prob), 0, max_log_prob)

        log_density = finite_max_log_prob.flatten() + \
                      np.log(np.sum(np.exp(log_prob_norm - finite_max_log_prob), axis=1) + 1e-40) # Add epsilon for log(0)

        # If original max_log_prob was -inf, the log_density should also be -inf
        log_density[np.isinf(max_log_prob.flatten())] = -np.inf

        return log_density

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability of each component given the data."""
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        responsibilities, _ = self._e_step(X) # _e_step already handles normalization
        return responsibilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the hardest assignment to the mixture components for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)

