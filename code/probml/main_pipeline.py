# -*- coding: utf-8 -*-
"""
Main analysis pipeline for Semi-Supervised MIWAE.
"""
from probml.core.utils import get_logger
from probml.preprocessing.data_preprocessor import DataPreprocessor
from probml.analysis.feature_selection import FeatureSelector, FeatureSelectorConfig
from probml.analysis.orchestrator import AnalysisOrchestrator
from probml.models.vae_base import VariationalAutoencoderBase
from probml.models.miwae import SemiSupMIWAE
from probml.models.trainer import VAETrainer
from probml.models.vae_helpers import prepare_vae_input, create_vae_from_artifacts
from probml.data_loading import load_nyc_data

import math
import os
import pickle
import time
import logging
from typing import List, Optional, Dict, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
# PyTorch Imports for VAE Dataset/Loader
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split

# ML Model Imports are removed as classification/regression is handled by VAE now
from sklearn.preprocessing import StandardScaler # Keep for potential use outside VAE prep


# --- Main Analysis Script Logic ---
if __name__ == "__main__":
    # --- Basic Setup ---
    main_logger = get_logger("MainRealEstateAnalysis_MIWAE", verbose=True)
    main_logger.info("================================================================")
    main_logger.info("==== Robust Real Estate Analysis - Semi-Supervised MIWAE ====")
    main_logger.info("================================================================")
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    main_logger.info(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # --- Configuration Parameters ---
    main_logger.info("--- Configuring Analysis Parameters ---")
    # Data Paths
    PARQUET_PATH = '/content/drive/MyDrive/e6691_2025Spring_nyre_local/data/processed_building_ml.parquet'
    CSV_PATH = 'building_ml_merged.csv' # Fallback

    # Core Columns
    PRICE_COL = 'sale_price' # Original scale price column name used by FS and Orchestrator
    LOG_PRICE_COL = 'log_sale_price' # Log-transformed price column (target for VAE Y head)
    DATE_COL_FOR_SPLIT = 'sale_date' # Used for time series validation if enabled

    # General Settings
    N_JOBS = -1 # For parallel processing where applicable
    VERBOSE_LOGGING = True

    # Feature Selector Config
    FS_AUTO_K_METHOD = "coverage"
    FS_TARGET_K = None # Set integer K if not using auto_k_method
    FS_CORRELATION_THRESHOLD = 0.9
    FS_COVERAGE_THRESHOLD = 0.9
    FS_METHODS_TO_RUN = ['variance', 'laplacian', 'pseudo_label']
    main_logger.info(f"Feature Selector Config: AutoK='{FS_AUTO_K_METHOD}', TargetK={FS_TARGET_K}, CorrThresh={FS_CORRELATION_THRESHOLD}, CovThresh={FS_COVERAGE_THRESHOLD}")

    # Analysis Orchestrator & Data Processing Config
    MAX_TIME_PIPELINE_SECONDS = 3600 * 2 # Max time for orchestrator
    MAX_CORE_ANALYSIS_SAMPLES = 20000 # Limit samples for computationally intensive analysis within orchestrator
    RUN_BOOTSTRAP_ORCHESTRATOR = False # Run bootstrapping in orchestrator?
    N_BOOTSTRAP_ITER = 30
    APPLY_SPATIAL = False # Apply spatial analysis in orchestrator?
    AUTO_TRANSFORM_SKEWED = True # Transform skewed features in orchestrator's preprocessor?
    TRANSFORM_TYPE = 'quantile' # Default transform type
    TRANSFORM_SKEW_THRESHOLD = 1.0 # Skewness threshold for transformation

    # Semi-Supervised MIWAE Training Config
    VAE_MODEL_TYPE = 'SemiSupMIWAE' # Model class to use
    VAE_LEARNING_RATE = 3e-4
    VAE_EPOCHS = 100
    VAE_KL_ANNEAL_EPOCHS = 30 # Epochs to linearly increase KLD weight (beta) to full value
    VAE_EARLY_STOPPING_PATIENCE = 15 # Patience for early stopping based on validation loss
    VAE_BATCH_SIZE = 512
    VAE_RECON_LOSS_X = 'mse' # Reconstruction loss type for X features ('mse' or 'huber')
    VAE_PRICE_LOSS_Y = 'gaussian_nll' # Loss type for Y target ('gaussian_nll' or 'mse')
    VAE_ALPHA_PRICE_LOSS = 10.0 # Weight (alpha) for the supervised price loss term (Y loss)
    main_logger.info(f"VAE Config: LR={VAE_LEARNING_RATE}, Epochs={VAE_EPOCHS}, Batch={VAE_BATCH_SIZE}, "
                     f"ReconLossX='{VAE_RECON_LOSS_X}', PriceLossY='{VAE_PRICE_LOSS_Y}', AlphaY={VAE_ALPHA_PRICE_LOSS}")

    # Prediction Config
    UNCERTAINTY_THRESHOLD = 1.0 # Threshold for filtering predictions based on log_std_dev

    # Validation Config
    USE_TIME_SERIES_VALIDATION = False # Use time-series split for VAE train/val? (Requires VAETrainer modifications not implemented here)
    N_TIME_SERIES_SPLITS = 5
    VAL_SPLIT_RATIO = 0.15 # Used for simple random train/val split if time series is off

    # Cross-validation config (non-stratified by year)
    N_FOLDS = 10                    # 10-way fold
    CV_SHUFFLE_SEED = RANDOM_SEED   # deterministic fold assignment

    # Output Files
    RESULTS_HISTORY_PKL = 'miwae_analysis_results.pkl' # For saving summary dict
    SAVED_VAE_MODEL_PATH = "miwae_final_model.pth" # Path to save final VAE model/trainer state
    SAVED_PREDICTIONS_PATH = "final_data_with_miwae_predictions.parquet" # Path for df with predictions
    FAILED_ORCHESTRATOR_PKL = "FAILED_orchestrator_results.pkl" # File if orchestrator fails

    # --- Step 1: Load Data ---
    main_logger.info(f"--- [Step 1/9] Loading Data ---")
    main_logger.info(f"Attempting to load Parquet: {PARQUET_PATH}")
    df = load_nyc_data(file_path=PARQUET_PATH, logger_instance=main_logger)
    if df is None:
        main_logger.warning(f"Parquet failed, trying CSV: {CSV_PATH}")
        df = load_nyc_data(file_path=CSV_PATH, logger_instance=main_logger)
    if df is None:
        main_logger.critical("Data loading failed from both Parquet and CSV paths. Halting.")
        raise FileNotFoundError(f"Could not load data from {PARQUET_PATH} or {CSV_PATH}")
    main_logger.info(f"Data loaded successfully. Initial shape: {df.shape}. Memory usage: {df.memory_usage(index=True, deep=True).sum() / (1024**2):.2f} MB")
    main_logger.debug(f"Initial columns: {df.columns.tolist()}")

    # --- Step 2: Define and Verify Price Column ---
    main_logger.info(f"--- [Step 2/9] Verifying Price Column ('{PRICE_COL}') ---")
    if PRICE_COL not in df.columns:
        fallback_price_col = 'building_sales_mean' # Example fallback
        if fallback_price_col in df.columns:
            df[PRICE_COL] = df[fallback_price_col]
            main_logger.warning(f"Original price column '{PRICE_COL}' not found. Using fallback column '{fallback_price_col}' as '{PRICE_COL}'.")
        else:
            main_logger.critical(f"Required price column '{PRICE_COL}' (and fallback '{fallback_price_col}') not found in DataFrame. Cannot proceed.")
            raise KeyError(f"Price column '{PRICE_COL}' not found.")
    else:
        main_logger.info(f"Using column '{PRICE_COL}' as the original scale price.")
    # Ensure log price column doesn't exist yet if it shouldn't (it will be created by orchestrator/preprocessor)
    if LOG_PRICE_COL in df.columns:
        main_logger.warning(f"Log-price column '{LOG_PRICE_COL}' already exists in input data. It will likely be overwritten by preprocessing steps.")

    # --- Step 3: Feature Selection ---
    main_logger.info(f"--- [Step 3/9] Performing Feature Selection (Targeting based on '{PRICE_COL}') ---")
    top_features_for_analysis: List[str] = []
    try:
        # Define ID/non-feature columns to ignore during selection
        # Note: PRICE_COL and LOG_PRICE_COL are explicitly excluded here
        id_cols_to_ignore_fs = list(set([
            'bbl', 'borough', 'block', 'lot', 'zipcode', 'address',
            'sale_date', PRICE_COL, LOG_PRICE_COL, # Exclude targets and date explicitly
            'xcoord','ycoord', 'latitude','longitude',
            # Various administrative/geographic IDs (add more specific ones if needed)
            'cd', 'council', 'schooldist', 'healtharea', 'policeprct', 'firecomp',
            'sanitboro', 'sanitdistrict', 'sanitsub', 'ct2010', 'tract2010',
            'nta', 'bct2020', 'bctcb2020', 'cb2000', 'firm07_flag', 'pfirm15_flag',
            'mappluto_f', 'plutomapid', 'version', 'notes', 'sanitdistr',
            'index', 'id', 'rowid', 'gid', 'uniqueid', 'parcelid', 'parid', 'parcelnumb',
            'condono', 'appbbl', 'tbl',
            # Columns potentially created later (shouldn't affect FS if run first)
            'has_positive_price', 'price_band', 'is_price_outlier', 'sample_weight'
        ]))
        id_cols_to_ignore_fs = [col for col in id_cols_to_ignore_fs if col in df.columns] # Keep only existing columns
        main_logger.debug(f"Columns ignored by Feature Selector: {id_cols_to_ignore_fs}")

        fs_config = FeatureSelectorConfig( # Use actual constructor if available
                variance_threshold=1e-4, correlation_threshold=FS_CORRELATION_THRESHOLD,
                methods_to_run=FS_METHODS_TO_RUN, coverage_threshold=FS_COVERAGE_THRESHOLD,
                bootstrap_B=30, bootstrap_k_per_iter=15 # Add other params as needed by your FS class
        )
        selector = FeatureSelector(
            df.copy(), # Use a copy for safety
            id_cols=id_cols_to_ignore_fs,
            target_col=PRICE_COL, # FS targets the original price column
            random_state=RANDOM_SEED, verbose=VERBOSE_LOGGING, n_jobs=N_JOBS, config=fs_config
        )

        # Execute selection based on configuration
        if FS_AUTO_K_METHOD:
            main_logger.info(f"Running FS with auto_k_method='{FS_AUTO_K_METHOD}'...")
            top_features_for_analysis = selector.select(auto_k_selection_method=FS_AUTO_K_METHOD)
        elif FS_TARGET_K is not None and FS_TARGET_K > 0:
            main_logger.info(f"Running FS targeting k={FS_TARGET_K} features...")
            top_features_for_analysis = selector.select(k=FS_TARGET_K)
        else:
            raise ValueError("Feature selection requires either FS_AUTO_K_METHOD or a positive FS_TARGET_K.")

        if not top_features_for_analysis:
            main_logger.critical("Feature selection returned an empty list of features. Halting.")
            raise ValueError("Feature selection failed to identify any features.")

        main_logger.info(f"Feature selection complete. Selected {len(top_features_for_analysis)} features for subsequent analysis.")
        main_logger.debug(f"Selected features: {top_features_for_analysis}")

    except Exception as e:
        main_logger.error(f"Feature selection process failed critically: {e}", exc_info=True)
        raise # Re-raise the exception to halt execution

    # --- Step 4: Analysis Orchestration (Preprocessing, Clustering, Dim Reduction Insights) ---
    main_logger.info(f"--- [Step 4/9] Running Analysis Orchestrator ---")
    main_logger.info(f"Orchestrator PriceCol: '{PRICE_COL}', LogPriceCol: '{LOG_PRICE_COL}', MaxSamples: {MAX_CORE_ANALYSIS_SAMPLES}")
    df_processed : Optional[pd.DataFrame] = None # Initialize
    main_artifacts : Dict[str, Any] = {}     # Initialize
    try:
        orchestrator_instance = AnalysisOrchestrator(
            random_state=RANDOM_SEED, verbose=VERBOSE_LOGGING, n_jobs=N_JOBS,
            max_total_time_seconds=MAX_TIME_PIPELINE_SECONDS,
            price_col=PRICE_COL, log_price_col_name=LOG_PRICE_COL, # Pass both price column names
            apply_spatial_analysis=APPLY_SPATIAL,
            min_price_percentile=0.005, max_price_percentile=0.995, # Example percentiles for internal price processing
            max_samples_limit=MAX_CORE_ANALYSIS_SAMPLES,
            max_bootstraps=N_BOOTSTRAP_ITER,
            default_transform_type=TRANSFORM_TYPE, # Pass transform settings
            transform_skew_threshold=TRANSFORM_SKEW_THRESHOLD,
            # Add other orchestrator specific parameters here based on its definition
        )
        analysis_run_output = orchestrator_instance.full_analysis_pipeline(
            df_input=df.copy(), # Pass original df
            numeric_features_for_analysis=top_features_for_analysis, # Pass selected features
            run_bootstrap=RUN_BOOTSTRAP_ORCHESTRATOR,
            max_samples_for_core_analysis=MAX_CORE_ANALYSIS_SAMPLES,
        )

        # --- Check Orchestrator Output ---
        run_status = analysis_run_output.get("status", "unknown")
        if run_status != "completed":
            error_msg = analysis_run_output.get("error", "Unknown orchestrator error")
            main_logger.error(f"Analysis Orchestrator failed! Status: '{run_status}', Error: '{error_msg}'.")
            main_logger.error(f"Keys available in orchestrator output: {list(analysis_run_output.keys())}")
            # Save partial results for debugging
            try:
                results_to_save = {"orchestrator_output": analysis_run_output, "selected_features": top_features_for_analysis}
                with open(FAILED_ORCHESTRATOR_PKL, 'wb') as f: pickle.dump(results_to_save, f)
                main_logger.info(f"Saved partial orchestrator results to '{FAILED_ORCHESTRATOR_PKL}'.")
            except Exception as e_save: main_logger.error(f"Failed to save partial orchestrator results: {e_save}")
            raise RuntimeError(f"Analysis Orchestrator failed: {error_msg}") # Halt execution

        # --- Extract Processed Data and Artifacts ---
        # **Important**: Adjust key based on actual orchestrator output structure
        # Assuming 'df_processed_and_rebalanced' holds the final DF state intended for VAE
        df_processed = analysis_run_output.get('df_processed_and_rebalanced')
        # Fallback to key from original script if above is not found
        if df_processed is None:
             df_processed = analysis_run_output.get('detailed_results_history', {}).get('df_fully_processed_before_sampling')

        main_artifacts = analysis_run_output.get("main_artifacts", {})

        if not isinstance(df_processed, pd.DataFrame) or df_processed.empty:
            main_logger.critical("Orchestrator completed but failed to return a valid processed DataFrame. Check orchestrator's results structure. Halting.")
            raise ValueError("Processed DataFrame ('df_processed_and_rebalanced' or similar key) not found or empty in orchestrator output.")

        main_logger.info(f"Analysis Orchestrator completed successfully. Processed DataFrame shape: {df_processed.shape}")
        main_logger.debug(f"Processed DataFrame columns: {df_processed.columns.tolist()}")
        main_logger.debug(f"Main artifacts keys: {list(main_artifacts.keys())}")

    except Exception as e_orch:
        main_logger.error(f"Analysis Orchestrator process failed critically: {e_orch}", exc_info=True)
        raise # Re-raise to halt execution

    # --- Step 5: Prepare Data for Semi-Supervised MIWAE ---
    main_logger.info(f"--- [Step 5/9] Preparing Data for {VAE_MODEL_TYPE} using '{LOG_PRICE_COL}' as target ---")

    # Define columns for VAE input: Should include selected X features + the LOG_PRICE_COL
    vae_input_feature_candidates = list(dict.fromkeys(top_features_for_analysis + [LOG_PRICE_COL]))
    # Verify these columns exist in the df_processed from orchestrator
    vae_input_feature_candidates = [f for f in vae_input_feature_candidates if f in df_processed.columns]
    if LOG_PRICE_COL not in vae_input_feature_candidates:
         main_logger.critical(f"CRITICAL: Target column '{LOG_PRICE_COL}' is missing from df_processed BEFORE VAE preparation. Check Orchestrator's output DataFrame. Halting.")
         raise ValueError(f"Target column '{LOG_PRICE_COL}' missing from processed DataFrame.")

    main_logger.info(f"Columns passed to prepare_vae_input: {len(vae_input_feature_candidates)}. Target (Y) column: '{LOG_PRICE_COL}'.")
    main_logger.debug(f"Columns for prepare_vae_input: {vae_input_feature_candidates}")

    try:
        # Prepare X and Y for MIWAE
        vae_X_filled_np, vae_X_mask_np, \
        vae_y_filled_np, vae_y_mask_np, \
        vae_feature_names_x_out, _, _ = prepare_vae_input(
            df=df_processed.copy(),               # Processed DataFrame from the orchestrator
            feature_columns=vae_input_feature_candidates,  # All candidate columns (X features + LOG_PRICE_COL)
            target_col=LOG_PRICE_COL,               # Specify the log-price as the supervised target
            scaler_type='standard'                  # Apply StandardScaler to X
        )

        # Validate prepare_vae_input outputs
        if vae_X_filled_np is None or vae_X_mask_np is None:
            raise ValueError("VAE input preparation failed for features (X). Check prepare_vae_input logs.")
        if vae_y_filled_np is None or vae_y_mask_np is None:
            raise ValueError(f"VAE input preparation failed for target ({LOG_PRICE_COL}). Check prepare_vae_input logs.")
        if not vae_feature_names_x_out:
            raise ValueError("No X feature names returned by prepare_vae_input.")

        vae_feature_names_x = vae_feature_names_x_out  # Final list of X columns
        main_logger.info(
            f"VAE data prepared. X features: {len(vae_feature_names_x)}  "
            f"(shapes X_filled={vae_X_filled_np.shape}, X_mask={vae_X_mask_np.shape}; "
            f"Y_filled={vae_y_filled_np.shape}, Y_mask={vae_y_mask_np.shape})"
        )
        main_logger.debug(f"Actual X features for VAE: {vae_feature_names_x}")

        # Build TensorDataset for training
        full_vae_dataset = TensorDataset(
            torch.from_numpy(vae_X_filled_np).float(),
            torch.from_numpy(vae_X_mask_np).float(),
            torch.from_numpy(vae_y_filled_np).float(),
            torch.from_numpy(vae_y_mask_np).float(),
        )
        main_logger.info(f"Created MIWAE TensorDataset with {len(full_vae_dataset)} samples.")

        # --- 10-fold, non-stratified 80/10/10 split (indices only) ---
        N = len(full_vae_dataset)
        all_idx = np.arange(N)

        rng = np.random.default_rng(CV_SHUFFLE_SEED)
        rng.shuffle(all_idx)

        # split shuffled indices into 10 folds with near-equal sizes
        folds: List[np.ndarray] = np.array_split(all_idx, N_FOLDS)
        main_logger.info(f"Constructed {N_FOLDS} non-stratified folds. Fold sizes: {[len(f) for f in folds]}")

    except Exception as e_vae_prep:
        main_logger.error(f"VAE data preparation failed critically: {e_vae_prep}", exc_info=True)
        raise

    # --- Step 6: Initialize/Load VAE Model and Trainer ---
    main_logger.info(f"--- [Step 6/9] Initializing {VAE_MODEL_TYPE} Model and Trainer ---")
    vae_model: Optional[Union[VariationalAutoencoderBase, SemiSupMIWAE]] = None
    vae_trainer_instance: Optional[VAETrainer] = None
    try:
        #TODO: reconcile 2-dim reality against 3-dim visualization needs
        # vae_model = create_vae_from_artifacts(
        #     artifacts=main_artifacts,
        #     feature_order=vae_feature_names_x + [LOG_PRICE_COL], # Pass ONLY X features here
        #     target_col_name=LOG_PRICE_COL, # Pass Y target name for SemiSup logic
        #     alpha_price_loss=VAE_ALPHA_PRICE_LOSS, # Pass alpha for model init
        #     # Add overrides if needed: latent_dim_override=..., price_head_layer_sizes=... etc.
        #     device=torch.device(DEVICE)
        # )
        # force at least 3 latent dims so downstream PCA(3) can actually run
        suggested = main_artifacts.get("latent_dim_suggestion", 3)
        vae_model = create_vae_from_artifacts(
            artifacts=main_artifacts,
            feature_order=vae_feature_names_x + [LOG_PRICE_COL],
            target_col_name=LOG_PRICE_COL,
            alpha_price_loss=VAE_ALPHA_PRICE_LOSS,
            # â† override latent_dim here:
            latent_dim_override = max(3, suggested),
            device=torch.device(DEVICE)
        )
        if vae_model is None:
            raise RuntimeError(f"Failed to create {VAE_MODEL_TYPE} model using create_vae_from_artifacts.")

        # Instantiate Trainer with the created model
        vae_trainer_instance = VAETrainer(
            model=vae_model,
            learning_rate=VAE_LEARNING_RATE,
            device=torch.device(DEVICE),
            verbose=VERBOSE_LOGGING,
            reconstruction_loss_type_x=VAE_RECON_LOSS_X, # Pass X loss type
            price_loss_type_y=VAE_PRICE_LOSS_Y,      # Pass Y loss type
            alpha_price_loss=VAE_ALPHA_PRICE_LOSS,   # Pass trainer's alpha (can override model's)
            kld_weight=1.0 # Base KLD weight before annealing
        )
        main_logger.info(f"{vae_model.__class__.__name__} model and VAETrainer initialized successfully.")

    except Exception as e_vae_init:
        main_logger.error(f"VAE model or trainer initialization failed: {e_vae_init}", exc_info=True)
        raise # Halt execution

    # --- [Step 7/9] 10-fold CV training with 80/10/10 (non-stratified) ---
    main_logger.info(f"--- [Step 7/9] Starting 10-fold CV: each run uses 80% train / 10% val / 10% test ---")

    # storage for stitched test predictions/uncertainties across folds
    stitched_mu_log = np.full((N,), np.nan, dtype=np.float64)
    stitched_logvar = np.full((N,), np.nan, dtype=np.float64)

    fold_histories: List[Dict[str, Any]] = []
    train_history = {} # Use this to store the list of fold histories
    vae_model_trained = None  # updated per fold

    for fold_id in range(N_FOLDS):
        test_idx = folds[fold_id]
        val_idx  = folds[(fold_id + 1) % N_FOLDS]
        train_idx = np.setdiff1d(all_idx, np.concatenate([test_idx, val_idx]), assume_unique=False)

        train_vae_dataset = Subset(full_vae_dataset, train_idx.tolist())
        val_vae_dataset   = Subset(full_vae_dataset, val_idx.tolist())
        test_vae_dataset  = Subset(full_vae_dataset, test_idx.tolist())

        main_logger.info(
            f"[Fold {fold_id+1}/{N_FOLDS}] sizes â†’ "
            f"Train={len(train_idx)} ({len(train_idx)/N:.1%}), "
            f"Val={len(val_idx)} ({len(val_idx)/N:.1%}), "
            f"Test={len(test_idx)} ({len(test_idx)/N:.1%})"
        )

        # fresh trainer each fold (reuse the already-constructed model class & artifacts)
        vae_model = create_vae_from_artifacts(
            artifacts=main_artifacts,
            feature_order=vae_feature_names_x + [LOG_PRICE_COL],
            target_col_name=LOG_PRICE_COL,
            alpha_price_loss=VAE_ALPHA_PRICE_LOSS,
            latent_dim_override=max(3, main_artifacts.get("latent_dim_suggestion", 3)),
            device=torch.device(DEVICE)
        )
        if vae_model is None:
            raise RuntimeError(f"Fold {fold_id+1}: failed to create {VAE_MODEL_TYPE} model.")

        vae_trainer_instance = VAETrainer(
            model=vae_model,
            learning_rate=VAE_LEARNING_RATE,
            device=torch.device(DEVICE),
            verbose=VERBOSE_LOGGING,
            reconstruction_loss_type_x=VAE_RECON_LOSS_X,
            price_loss_type_y=VAE_PRICE_LOSS_Y,
            alpha_price_loss=VAE_ALPHA_PRICE_LOSS,
            kld_weight=1.0
        )

        # train with early stopping on this fold's val set
        history = vae_trainer_instance.train(
            train_dataset=train_vae_dataset,
            val_dataset=val_vae_dataset,
            epochs=VAE_EPOCHS,
            batch_size=VAE_BATCH_SIZE,
            kld_anneal_epochs=VAE_KL_ANNEAL_EPOCHS,
            early_stopping_patience=VAE_EARLY_STOPPING_PATIENCE
        )
        fold_histories.append(history)
        vae_model_trained = vae_trainer_instance.model # Keep the last trained model

        # --- predict on this fold's test set and stitch into global arrays ---
        X_full = full_vae_dataset.tensors[0]  # [N, d]
        X_test = X_full[test_idx].to(DEVICE)

        if hasattr(vae_trainer_instance, 'predict_price_and_uncertainty'):
            log_mu, log_var = vae_trainer_instance.predict_price_and_uncertainty(
                X_test, batch_size=VAE_BATCH_SIZE * 2
            )
            if (log_mu is None) or (log_var is None):
                main_logger.error(f"[Fold {fold_id+1}] Prediction returned None.")
            else:
                stitched_mu_log[test_idx] = log_mu.reshape(-1)
                stitched_logvar[test_idx] = log_var.reshape(-1)
                main_logger.info(f"[Fold {fold_id+1}] Stored test predictions for {len(test_idx)} rows.")
        else:
            main_logger.error("VAETrainer missing 'predict_price_and_uncertainty'. Cannot predict.")

    train_history = {"fold_histories": fold_histories} # Store all histories
    main_logger.info(f"VAE 10-fold CV training finished.")


    # --- [Step 8/9] Stitched test-fold predictions across CV runs ---
    main_logger.info(f"--- [Step 8/9] Consolidating stitched predictions from 10-fold CV ---")
    df_for_predictions = df_processed.copy()

    # Convert log(mu), logvar â†’ original scale + log-scale std
    # Note: model emits predictions in raw log(price) space
    pred_ok_mask = ~np.isnan(stitched_mu_log)
    num_pred = int(pred_ok_mask.sum())

    if num_pred == 0:
        main_logger.warning("No stitched predictions found. Skipping consolidation.")
    else:
        # Get predictions for rows where prediction was successful
        valid_log_mu = stitched_mu_log[pred_ok_mask]
        valid_log_var = stitched_logvar[pred_ok_mask]

        log_std = np.sqrt(np.exp(valid_log_var))
        price_pred = np.maximum(0.0, np.exp(valid_log_mu)) # Ensure non-negative

        pred_price_col_name = f"cv_predicted_{PRICE_COL}"
        pred_std_col_name   = f"cv_pred_uncertainty_{LOG_PRICE_COL}_std"
        pred_flag_col_name  = "cv_prediction_available"
        pred_ok_col_name = "prediction_uncertainty_ok" # Keep this name for consistency

        # init as NaN/False
        df_for_predictions[pred_price_col_name] = np.nan
        df_for_predictions[pred_std_col_name]   = np.nan
        df_for_predictions[pred_flag_col_name]  = False
        df_for_predictions[pred_ok_col_name]    = False # Init as False

        # Fill in the valid predictions
        df_for_predictions.loc[pred_ok_mask, pred_price_col_name] = price_pred
        df_for_predictions.loc[pred_ok_mask, pred_std_col_name]   = log_std
        df_for_predictions.loc[pred_ok_mask, pred_flag_col_name]  = True

        # Calculate uncertainty flag only for rows with predictions
        df_for_predictions.loc[pred_ok_mask, pred_ok_col_name] = \
            (df_for_predictions.loc[pred_ok_mask, pred_std_col_name] < UNCERTAINTY_THRESHOLD)

        ok_under_thresh = (df_for_predictions[pred_ok_col_name]).sum()
        total_count = len(df_for_predictions)
        main_logger.info(
            f"Stitched predictions filled for {num_pred}/{total_count} rows "
            f"({num_pred/total_count:.1%})."
        )
        main_logger.info(f"Predictions within uncertainty threshold ({UNCERTAINTY_THRESHOLD}): {ok_under_thresh}/{num_pred} ({ok_under_thresh/num_pred*100:.1f}% of predicted)")
        main_logger.debug(f"Prediction summary:\n{df_for_predictions[[pred_price_col_name, pred_std_col_name]].describe()}")


    # --- Step 9: Save Final Results ---
    main_logger.info(f"--- [Step 9/9] Saving Final Results ---")
    try:
        # Save summary dictionary (excluding large objects)
        results_summary = {
            'orchestrator_summary': analysis_run_output.get('summary_report', {}),
            'orchestrator_artifacts_summary': {k: v for k, v in main_artifacts.items() if not isinstance(v, (pd.DataFrame, np.ndarray, StandardScaler, torch.nn.Module))},
            'selected_features_for_analysis': top_features_for_analysis, # Features selected by FS
            'vae_actual_x_features_used': vae_feature_names_x, # Actual X features input to VAE
            'vae_target_y_column': LOG_PRICE_COL if vae_model_trained and isinstance(vae_model_trained, SemiSupMIWAE) else None,
            'vae_training_history': train_history, # This now contains fold_histories
            'config_summary': { # Log key run parameters
                'PRICE_COL': PRICE_COL, 'LOG_PRICE_COL': LOG_PRICE_COL,
                'FS_AUTO_K_METHOD': FS_AUTO_K_METHOD, 'FS_TARGET_K': FS_TARGET_K, 'FS_COVERAGE_THRESHOLD': FS_COVERAGE_THRESHOLD,
                'MAX_CORE_ANALYSIS_SAMPLES': MAX_CORE_ANALYSIS_SAMPLES, 'RUN_BOOTSTRAP_ORCHESTRATOR': RUN_BOOTSTRAP_ORCHESTRATOR,
                'VAE_MODEL_TYPE': VAE_MODEL_TYPE, 'VAE_ALPHA_PRICE_LOSS': VAE_ALPHA_PRICE_LOSS, 'VAE_EPOCHS': VAE_EPOCHS,
                'VAE_RECON_LOSS_X': VAE_RECON_LOSS_X, 'VAE_PRICE_LOSS_Y': VAE_PRICE_LOSS_Y,
                'UNCERTAINTY_THRESHOLD': UNCERTAINTY_THRESHOLD,
                'N_FOLDS_CV': N_FOLDS # Add CV info
            }
        }
        with open(RESULTS_HISTORY_PKL, 'wb') as f:
            pickle.dump(results_summary, f)
        main_logger.info(f"Summary results dictionary saved to '{RESULTS_HISTORY_PKL}'")

        # Save final DataFrame with predictions if they exist
        pred_col_name_check = f"cv_predicted_{PRICE_COL}" # Check for the CV prediction column
        if pred_col_name_check in df_for_predictions.columns:
            try:
                df_for_predictions.to_parquet(SAVED_PREDICTIONS_PATH)
                main_logger.info(f"DataFrame with predictions saved to '{SAVED_PREDICTIONS_PATH}'")
            except Exception as e_save_df:
                main_logger.error(f"Failed to save DataFrame with predictions to Parquet: {e_save_df}")
                try: # Fallback to CSV
                    df_for_predictions.to_csv(SAVED_PREDICTIONS_PATH.replace('.parquet','.csv'), index=False)
                    main_logger.info(f"DataFrame with predictions saved to CSV fallback: '{SAVED_PREDICTIONS_PATH.replace('.parquet','.csv')}'")
                except Exception as e_save_csv:
                    main_logger.error(f"Failed to save DataFrame with predictions to CSV fallback: {e_save_csv}")
        else:
            main_logger.warning("No CV prediction columns found in the final DataFrame. Skipping save of prediction DataFrame.")

    except Exception as e_final_save:
        main_logger.error(f"Error occurred during final saving steps: {e_final_save}", exc_info=True)

    main_logger.info(f"==== Main Analysis Script ({VAE_MODEL_TYPE} with {N_FOLDS}-Fold CV) Finished ====")
    main_logger.info("=================================================================================")


