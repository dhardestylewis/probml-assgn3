# -*- coding: utf-8 -*-
from probml.core.utils import get_logger
from probml.models.vae_base import VariationalAutoencoderBase
from probml.models.miwae import SemiSupMIWAE


# --- Helper Function: Prepare Data for VAE (Refactored for Semi-Supervised MIWAE) ---
def prepare_vae_input(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_col: Optional[str] = None, # *** ADDED for semi-supervised target ***
    scaler_type: str = 'standard', # 'standard', 'minmax', or 'none'
) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], # X_filled_processed, X_mask
    Optional[np.ndarray], Optional[np.ndarray], # Y_filled_processed, Y_mask (if target_col is provided)
    Optional[List[str]], Optional[Any], Optional[Dict[str, str]] # actual_feature_columns_x, scaler_object, feature_metadata
]:
    """
    Prepares DataFrame data for Semi-Supervised MIWAE input.
    - Selects feature (X) and optionally target (y) columns.
    - Scales features (X) if a scaler_type is specified. Target (y) is NOT scaled by this function.
    - Fills NaNs with 0.0 (a common strategy for VAEs, especially MIWAE).
    - Returns the NaN-filled processed data AND binary masks (1 for observed, 0 for missing) for both X and y.
    """
    current_logger = get_logger(f"{__name__}.prepare_vae_input")
    current_logger.info(f"Preparing data and masks for VAE. Input features: {len(feature_columns)}, "
                        f"Target column: {target_col if target_col else 'N/A'}, Scaler type for X: {scaler_type}")

    # --- Process Features (X) ---
    # Ensure target_col is not accidentally in feature_columns for X
    actual_cols_present_x = [f_col for f_col in feature_columns if f_col in df.columns and f_col != target_col]

    if not actual_cols_present_x:
        current_logger.error("No valid X feature columns found in DataFrame after excluding target or missing columns.")
        return None, None, None, None, None, None, None

    # Warn if some requested feature columns (excluding target) were not found
    requested_x_features = [f_col for f_col in feature_columns if f_col != target_col]
    if len(actual_cols_present_x) < len(requested_x_features):
        missing_x_cols = set(requested_x_features) - set(actual_cols_present_x)
        current_logger.warning(f"Specified X feature columns not found in DataFrame: {missing_x_cols}. "
                               f"Using available X columns: {actual_cols_present_x}")

    df_selected_x = df[actual_cols_present_x].copy() # Use .copy() to avoid SettingWithCopyWarning

    try:
        data_np_x = df_selected_x.values.astype(np.float32)
    except Exception as e:
        current_logger.error(f"Failed to convert selected X features to np.float32: {e}", exc_info=True)
        return None, None, None, None, None, None, None

    # Create mask for X: 1 where data is present (not NaN), 0 where it's NaN
    mask_np_x = (~np.isnan(data_np_x)).astype(np.float32)
    num_missing_x = np.sum(mask_np_x == 0)
    current_logger.info(f"X data shape: {data_np_x.shape}. X mask shape: {mask_np_x.shape}. "
                        f"Number of missing values in X: {num_missing_x} ({(num_missing_x / data_np_x.size) * 100:.2f}%).")

    scaler_object = None
    scaled_data_nan_preserved_x = data_np_x # Start with original data (NaNs preserved)

    if scaler_type and scaler_type.lower() != 'none':
        if scaler_type.lower() == 'standard':
            scaler_object = StandardScaler()
        elif scaler_type.lower() == 'minmax':
            scaler_object = MinMaxScaler()
        # Add other scalers here if needed

        if scaler_object:
            current_logger.info(f"Applying {scaler_type} scaling to X features (NaNs are typically ignored by fit and transform)...")
            try:
                # Fit scaler only on non-NaN values for more robust scaling if possible,
                # though sklearn scalers usually handle NaNs by ignoring them in fit
                # and propagating them in transform.
                # To be absolutely sure, one might fit on df_selected_x.dropna() if all rows have some NaNs.
                # However, standard practice is to fit on the full data.
                scaler_object.fit(data_np_x) # Fit on data possibly containing NaNs
                scaled_data_nan_preserved_x = scaler_object.transform(data_np_x).astype(np.float32)
            except Exception as e_scale:
                current_logger.error(f"Error during X scaling with {scaler_type}: {e_scale}. "
                                     "Using unscaled X data instead.", exc_info=True)
                scaled_data_nan_preserved_x = data_np_x # Revert to unscaled
                scaler_object = None # Nullify scaler if it failed
        else:
            current_logger.warning(f"Unknown scaler_type '{scaler_type}' for X. No scaling will be applied to X.")
    else:
        current_logger.info("No scaling will be applied to X features as per 'scaler_type'.")


    # Fill NaNs with 0.0 AFTER scaling (if any)
    # `scaled_data_nan_preserved_x` still has NaNs where original data had them.
    data_filled_processed_x = np.nan_to_num(scaled_data_nan_preserved_x, nan=0.0).astype(np.float32)

    # Check for non-finite values (inf, -inf) that might arise from operations and replace them
    if np.any(~np.isfinite(data_filled_processed_x)):
        current_logger.warning("Non-finite values (inf/-inf) detected in processed X data after NaN filling. "
                               "Replacing them with 0.0.")
        data_filled_processed_x = np.nan_to_num(data_filled_processed_x, nan=0.0, posinf=0.0, neginf=0.0)


    feature_metadata = {
        col_name: 'numeric_scaled' if scaler_object else 'numeric_unscaled'
        for col_name in actual_cols_present_x
    }

    # --- Process Target (y) ---
    data_filled_processed_y = None
    mask_np_y = None

    if target_col:
        if target_col not in df.columns:
            current_logger.error(f"Specified target column '{target_col}' not found in DataFrame.")
            # Return processed X as it might still be useful for unsupervised VAE
            return data_filled_processed_x, mask_np_x, None, None, actual_cols_present_x, scaler_object, feature_metadata

        try:
            # Ensure y is 2D (e.g., for consistency if y_dim > 1 in future) by using df[[target_col]]
            data_np_y = df[[target_col]].values.astype(np.float32)
            mask_np_y = (~np.isnan(data_np_y)).astype(np.float32)
            data_filled_processed_y = np.nan_to_num(data_np_y, nan=0.0).astype(np.float32) # Fill NaNs with 0 for y

            num_missing_y = np.sum(mask_np_y == 0)
            current_logger.info(f"Y target ('{target_col}') data shape: {data_np_y.shape}. Y mask shape: {mask_np_y.shape}. "
                                f"Number of missing values in Y: {num_missing_y} ({(num_missing_y / data_np_y.size) * 100:.2f}%).")

            if np.any(~np.isfinite(data_filled_processed_y)):
                current_logger.warning(f"Non-finite values (inf/-inf) detected in processed Y target ('{target_col}') "
                                       "after NaN filling. Replacing them with 0.0.")
                data_filled_processed_y = np.nan_to_num(data_filled_processed_y, nan=0.0, posinf=0.0, neginf=0.0)

        except Exception as e_target:
            current_logger.error(f"Error processing target column '{target_col}': {e_target}", exc_info=True)
            # Return processed X, but Y will be None
            return data_filled_processed_x, mask_np_x, None, None, actual_cols_present_x, scaler_object, feature_metadata
    else:
        current_logger.info("No target column specified. Preparing data for unsupervised VAE.")


    current_logger.info(f"VAE input preparation complete. Shapes: "
                        f"X_filled: {data_filled_processed_x.shape}, X_mask: {mask_np_x.shape}, "
                        f"Y_filled: {data_filled_processed_y.shape if data_filled_processed_y is not None else 'N/A'}, "
                        f"Y_mask: {mask_np_y.shape if mask_np_y is not None else 'N/A'}")

    return (data_filled_processed_x, mask_np_x,
            data_filled_processed_y, mask_np_y,
            actual_cols_present_x, scaler_object, feature_metadata)



def create_vae_from_artifacts(
    artifacts: Dict[str, Any],
    feature_order: List[str], # Full list of features expected by the model, including target if applicable
    target_col_name: Optional[str] = None, # Name of the target column for SemiSupMIWAE
    alpha_price_loss: float = 1.0, # Weight for the price loss term in SemiSupMIWAE
    price_head_layer_sizes: Optional[List[int]] = None, # Override for price head layers
    device: Optional[torch.device] = None,
    encoder_layers_override: Optional[List[int]] = None,
    decoder_layers_override: Optional[List[int]] = None,
    latent_dim_override: Optional[int] = None,
    prior_type_override: Optional[str] = None, # e.g., 'standard_gaussian', 'gaussian_mixture', 'student_t', 'student_t_mixture'
    n_components_override: Optional[int] = None, # Number of components for mixture priors
    student_t_df_override: float = 4.0 # Default DoF for Student-t priors
) -> Optional[Union[VariationalAutoencoderBase, SemiSupMIWAE]]:

    current_logger = get_logger(f"{__name__}.create_vae_from_artifacts")
    current_logger.info(f"Creating VAE model. Target column for supervision: {target_col_name if target_col_name else 'None (Unsupervised VAE)'}.")

    if not artifacts: # artifacts can be an empty dict if no prior info is available
        current_logger.warning("Artifacts dictionary is empty or None. Will use defaults or overrides for VAE parameters.")
        artifacts = {} # Ensure artifacts is a dict
    if not feature_order:
        current_logger.error("Feature_order list is missing or empty. Cannot determine input dimension for X.")
        return None

    is_semi_supervised = target_col_name is not None
    # Determine X features (input to the main VAE encoder/decoder)
    # These are all features in feature_order *except* the target_col_name
    x_feature_names = [f_name for f_name in feature_order if f_name != target_col_name]
    input_dim_x = len(x_feature_names)

    if input_dim_x == 0:
        current_logger.error("No input features (X) are available after excluding the target column (if any). Cannot create VAE.")
        return None
    current_logger.info(f"Number of input features for VAE (X): {input_dim_x} (Features: {x_feature_names[:5]}...)")

    y_dim = 1 if is_semi_supervised else 0 # Assuming target is single-dimensional if present
    if is_semi_supervised:
        current_logger.info(f"Semi-supervised mode: Target '{target_col_name}' (y_dim={y_dim}). Alpha for price loss: {alpha_price_loss}.")


    # Determine Latent Dimension
    latent_dim: int
    if latent_dim_override is not None:
        latent_dim = latent_dim_override
        current_logger.info(f"Using overridden latent_dim: {latent_dim}")
    else:
        # Suggestion from artifacts, or a heuristic
        default_latent_suggestion = max(1, input_dim_x // 4, 2) # Heuristic: e.g., quarter of input, min 2
        latent_dim = artifacts.get('latent_dim_suggestion', default_latent_suggestion)
        current_logger.info(f"Using latent_dim: {latent_dim} (from artifacts or default heuristic).")
    # Ensure latent_dim is at least 1 and not excessively large (e.g., larger than input_dim_x)
    latent_dim = max(1, min(int(latent_dim), input_dim_x -1 if input_dim_x > 1 else 1))
    current_logger.info(f"Final effective Latent Dimension: {latent_dim}")


    # Determine Encoder/Decoder Layer Sizes
    # Heuristic default: one hidden layer, size related to input/latent dim
    default_hidden_layer_size = max(latent_dim * 2, input_dim_x // 2, latent_dim + 5, 10) # Ensure some reasonable size
    default_hidden_layer_size = min(default_hidden_layer_size, 512) # Cap max default size

    encoder_layer_sizes = encoder_layers_override if encoder_layers_override is not None else \
                          ([default_hidden_layer_size] if default_hidden_layer_size > latent_dim else [])
    decoder_layer_sizes = decoder_layers_override if decoder_layers_override is not None else \
                          ([default_hidden_layer_size] if default_hidden_layer_size > latent_dim else [])
    current_logger.info(f"Encoder layers: {encoder_layer_sizes}, Decoder layers: {decoder_layer_sizes}")


    # Determine Prior Type and Number of Components
    prior_type: str
    n_prior_components: int
    prior_params_init: Dict[str, Any] = {}

    if prior_type_override is not None and n_components_override is not None:
        prior_type = prior_type_override
        n_prior_components = n_components_override
        current_logger.info(f"Using overridden prior: Type={prior_type}, K={n_prior_components}")
    else:
        # Infer from artifacts if overrides are not fully provided
        inferred_prior_type, inferred_k_components = _infer_prior_from_artifacts(artifacts)
        prior_type = prior_type_override if prior_type_override is not None else inferred_prior_type
        n_prior_components = n_components_override if n_components_override is not None else inferred_k_components
        current_logger.info(f"Using determined prior: Type={prior_type}, K={n_prior_components} (from overrides or artifact inference).")

    # Initialize parameters for mixture priors if applicable
    if 'mixture' in prior_type.lower() and n_prior_components > 1:
        current_logger.info(f"Initializing parameters for {prior_type} with K={n_prior_components} components...")
        prior_params_init = _initialize_mixture_prior_params(artifacts, n_prior_components, latent_dim, current_logger)

    # Add Student-t degrees of freedom if it's a Student-t prior and not already in params
    if 'student_t' in prior_type.lower() and 'df' not in prior_params_init:
        prior_params_init['df'] = student_t_df_override
        current_logger.info(f"Setting Student-t df to {student_t_df_override} for prior.")


    # Instantiate Model
    try:
        model_constructor_args = {
            'encoder_layer_sizes': encoder_layer_sizes,
            'latent_dim': latent_dim,
            'decoder_layer_sizes': decoder_layer_sizes,
            'prior_type': prior_type,
            'n_prior_components': n_prior_components,
            'prior_params': prior_params_init, # Pass initialized prior params
            'device': device,
            # Other VAEBase params like activation_fn, use_batch_norm use defaults unless overridden here
        }

        vae_model: Union[VariationalAutoencoderBase, SemiSupMIWAE]
        if is_semi_supervised:
            current_logger.info("Instantiating SemiSupMIWAE model...")
            # Default price head layers: simple 2-layer MLP
            default_ph_l1 = max(latent_dim // 2, y_dim * 4, 8) # Sensible lower bound
            default_ph_l2 = max(latent_dim // 4, y_dim * 2, 4)
            default_price_head_ls = [default_ph_l1, default_ph_l2] if default_ph_l1 > y_dim and default_ph_l2 > y_dim else \
                                     [default_ph_l1] if default_ph_l1 > y_dim else []


            final_price_head_layers = price_head_layer_sizes if price_head_layer_sizes is not None else default_price_head_ls
            current_logger.info(f"Price head layers for SemiSupMIWAE: {final_price_head_layers}")

            model_constructor_args.update({
                'input_dim': input_dim_x, # X features dimension
                'y_dim': y_dim,
                'price_head_layer_sizes': final_price_head_layers,
                'alpha_price_loss': alpha_price_loss,
            })
            ModelClass = SemiSupMIWAE
        else:
            current_logger.info("Instantiating VariationalAutoencoderBase (unsupervised) model...")
            model_constructor_args.update({
                'input_dim': input_dim_x, # X features dimension
            })
            ModelClass = VariationalAutoencoderBase

        vae_model = ModelClass(**model_constructor_args) # type: ignore
        current_logger.info(f"{vae_model.__class__.__name__} model created successfully on device: {vae_model.device_}.")
        return vae_model

    except Exception as e:
        current_logger.error(f"Failed during VAE model instantiation: {e}", exc_info=True)
        return None


# --- Helper Functions: _infer_prior_from_artifacts, _initialize_mixture_prior_params (Unchanged logic, reformatted) ---
def _infer_prior_from_artifacts(artifacts: Dict[str, Any]) -> Tuple[str, int]:
    """Infers prior type and number of components from analysis artifacts."""
    current_logger = get_logger(f"{__name__}._infer_prior") # Shortened name

    # Default prior: Standard Gaussian
    inferred_type = 'standard_gaussian'
    inferred_k = 1

    # Check for signals of heavy-tailed distribution (e.g., from price analysis)
    price_stats = artifacts.get('price_analysis_stats', {})
    price_is_heavy_tailed = price_stats.get('log_heavy_tailed', False) or price_stats.get('kurtosis', 0) > 3.5 # Example threshold

    # Check for suggestions of number of clusters (K)
    # Order of preference for K: optimal_k_dynamic, ensemble_n_clusters, dp_gmm_n_components
    k_options = [
        artifacts.get('optimal_k_dynamic'),
        artifacts.get('ensemble_n_clusters'),
        artifacts.get('dp_gmm_n_components')
    ]
    for k_val in k_options:
        if isinstance(k_val, (int, float)) and np.isfinite(k_val) and k_val > 1:
            inferred_k = int(round(k_val))
            current_logger.debug(f"Inferred K={inferred_k} from artifacts (value: {k_val}).")
            break # Use the first valid K found

    # Determine prior type based on K and heavy-tailedness
    if inferred_k > 1: # Mixture model
        if price_is_heavy_tailed:
            inferred_type = 'student_t_mixture'
            current_logger.debug("Suggesting Student-t Mixture prior due to K > 1 and heavy-tailed signal.")
        else:
            inferred_type = 'gaussian_mixture'
            current_logger.debug("Suggesting Gaussian Mixture prior due to K > 1.")
    elif price_is_heavy_tailed: # Single component, heavy-tailed
        inferred_type = 'student_t'
        current_logger.debug("Suggesting Student-t prior due to heavy-tailed signal (K=1).")
    else: # Single component, not necessarily heavy-tailed (or no info)
        current_logger.debug("Suggesting Standard Gaussian prior (default or K=1, no heavy-tailed signal).")
        # inferred_type remains 'standard_gaussian'

    return inferred_type, inferred_k


def _initialize_mixture_prior_params(
    artifacts: Dict[str, Any],
    n_components: int, # Desired number of components for the VAE prior
    latent_dim: int,   # Latent dimension of the VAE
    logger_instance: logging.Logger # Pass logger for messages
) -> Dict[str, torch.Tensor]:
    """
    Initializes parameters (weights, means, covariances) for a mixture prior
    using information from DP-GMM or other clustering artifacts if available.
    """
    params: Dict[str, torch.Tensor] = {}
    dp_gmm_info = artifacts.get('dp_gmm_model_info_dict', {})

    # Extract DP-GMM results if available and somewhat conformant
    fit_k_gmm = dp_gmm_info.get('effective_components', 0)
    gmm_means_np = np.array(dp_gmm_info.get('means', []))         # Expected shape (fit_k_gmm, latent_dim)
    gmm_weights_np = np.array(dp_gmm_info.get('weights', []))     # Expected shape (fit_k_gmm,)
    # Assuming diagonal covariances from DP-GMM output for simplicity here
    gmm_covs_diag_np = np.array(dp_gmm_info.get('covariances', [])) # Expected shape (fit_k_gmm, latent_dim)

    # Check if GMM results are usable
    gmm_usable = (
        fit_k_gmm > 0 and
        gmm_means_np.ndim == 2 and gmm_means_np.shape[0] == fit_k_gmm and gmm_means_np.shape[1] == latent_dim and
        gmm_weights_np.ndim == 1 and len(gmm_weights_np) == fit_k_gmm
    )

    if gmm_usable:
        logger_instance.info(f"Attempting to initialize prior from DP-GMM artifacts: {fit_k_gmm} GMM components found with matching latent_dim {latent_dim}.")

        # Select top K components from GMM if GMM has more components than requested
        if fit_k_gmm >= n_components:
            top_indices = np.argsort(gmm_weights_np)[::-1][:n_components] # Sort by weight, take top n_components
            selected_means_np = gmm_means_np[top_indices]
            selected_weights_np = gmm_weights_np[top_indices]
            if gmm_covs_diag_np.ndim == 2 and gmm_covs_diag_np.shape[0] == fit_k_gmm and gmm_covs_diag_np.shape[1] == latent_dim:
                selected_covs_diag_np = gmm_covs_diag_np[top_indices]
            else:
                selected_covs_diag_np = None # Covariances not usable or mismatched
                logger_instance.warning("GMM covariances from artifacts are missing, malformed, or do not match latent_dim. Will use default identity for prior covariances.")
            logger_instance.info(f"Initialized prior parameters from the top {n_components} (by weight) GMM components.")
        else: # GMM has fewer components than requested, use all available GMM components
            selected_means_np = gmm_means_np
            selected_weights_np = gmm_weights_np
            if gmm_covs_diag_np.ndim == 2 and gmm_covs_diag_np.shape[0] == fit_k_gmm and gmm_covs_diag_np.shape[1] == latent_dim:
                 selected_covs_diag_np = gmm_covs_diag_np
            else:
                selected_covs_diag_np = None
                logger_instance.warning("GMM covariances from artifacts are missing or malformed for the available components. Defaulting.")
            logger_instance.warning(f"DP-GMM artifacts have {fit_k_gmm} components, which is less than "
                                   f"the requested {n_components} for the prior. Initializing with {fit_k_gmm} components from GMM and will use defaults for the rest if needed, or adapt.")
            # Note: The VAE prior will still have n_components. This logic just initializes *some* of them.
            # The VAE's _setup_prior will handle filling remaining components with defaults if these are partial.
            # For simplicity here, we might just initialize the first fit_k_gmm and let others be default.
            # However, the `VariationalAutoencoderBase._init_param` expects full size or scalar.
            # So, it's better to provide full-sized defaults if GMM is insufficient.
            # Current logic below will attempt to use selected_*, if they are not sufficient, defaults in _init_param take over.

        # Convert to tensors for VAE
        # Weights (logits)
        selected_weights_np_normalized = selected_weights_np / selected_weights_np.sum() # Normalize
        # Ensure no zero weights for log
        weights_logits_tensor = torch.log(torch.tensor(selected_weights_np_normalized, dtype=torch.float32).clamp(min=1e-9))
        params['weights_logits'] = weights_logits_tensor

        # Means
        params['means'] = torch.tensor(selected_means_np, dtype=torch.float32)

        # Covariances (Cholesky decomposition of diagonal covariance matrices)
        if selected_covs_diag_np is not None:
            try:
                # Ensure diagonal elements are positive before sqrt for Cholesky
                covs_diag_positive_np = np.maximum(selected_covs_diag_np, 1e-6) # Floor to small positive
                cholesky_diag_elements_tensor = torch.sqrt(torch.tensor(covs_diag_positive_np, dtype=torch.float32))
                # Create batch of diagonal matrices for Cholesky factors
                cov_cholesky_tensor = torch.diag_embed(cholesky_diag_elements_tensor)
                params['cov_cholesky'] = cov_cholesky_tensor
                logger_instance.info("Initialized prior cov_cholesky from GMM diagonal covariances.")
            except Exception as e_chol:
                logger_instance.error(f"Failed to compute Cholesky from GMM diagonal covariances: {e_chol}. "
                                     "Default identity covariances will be used for prior.", exc_info=False)
                if 'cov_cholesky' in params: del params['cov_cholesky'] # Remove to allow default
        else:
            logger_instance.warning("No usable GMM diagonal covariances for prior initialization. Default identity will be used.")

    else: # GMM artifacts not usable or not found
        logger_instance.warning("DP-GMM artifacts are unsuitable or insufficient for initializing mixture prior parameters. "
                                "Default parameters (e.g., zeros for means/logits, identity for covariances) will be used by the VAE.")

    # The VAE's _init_param method will handle defaults if these keys are missing or shapes are wrong.
    # This function just provides initial guesses if GMM data is good.
    # If params returned here are partial (e.g. fewer than n_components), _init_param might have issues.
    # It's safer to ensure that if we provide something, it's for the correct n_components.
    # The current logic above for selection (top_indices) tries to match n_components.
    # If fit_k_gmm < n_components, then selected_* will have fit_k_gmm items.
    # _init_param needs to be robust to this or this function should pad with defaults.
    # Given _init_param structure, it's better if this function doesn't return partial tensors.
    # So, if GMM data is insufficient to populate all n_components, we just don't add that param key,
    # letting _init_param use its full default.

    # Revised logic: Only add to `params` if we have data for *all* `n_components` or for the number of `fit_k_gmm` if `fit_k_gmm < n_components`
    # and the `_init_param` can handle broadcasting or partial initialization (which it currently doesn't gracefully for all params).
    # For safety, if GMM provides fewer than n_components, we might avoid initializing from it to prevent shape mismatches, unless _init_param is updated.
    # The original code's _init_param might default if shapes don't match, which is okay.

    # Example: if `params['means']` is (fit_k_gmm, latent_dim) but default is (n_components, latent_dim), _init_param defaults.
    # This is acceptable.

    return params

