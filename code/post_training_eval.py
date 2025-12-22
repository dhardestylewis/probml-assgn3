# === POST-TRAINING EVAL + VIZ (robust to missing cv_* columns) ===

import os, glob, math, logging, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # only used in summary artifacts

# Optional: for QQ-plot; we guard against missing scipy.
try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    print("[Eval] WARNING: scipy not available; QQ-plot will be skipped.")

# ----------------------------------------------------------
# 0. Mount Drive (Colab) and basic setup
# ----------------------------------------------------------
try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Eval] Using device: {DEVICE}")

logger = logging.getLogger("MIWAE_Eval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

# Ensure plots render at actual size in Colab
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

# ----------------------------------------------------------
# 1. Paths (aligned with your AlphaScan cell)
# ----------------------------------------------------------
RESULTS_ROOT = "/content/drive/MyDrive/e6691_2025Spring_nyre_local/results"
FALLBACK_RUN_DIR = "/content/drive/MyDrive/e6691_2025Spring_nyre_local/results/2025-10-12_01-58-16"

RESULTS_PKL_NAME = "miwae_analysis_results.pkl"
PRED_PARQUET_NAME = "final_data_with_miwae_predictions.parquet"
PRED_CSV_NAME = "final_data_with_miwae_predictions.csv"

def _latest_run_dir(results_root: str, fallback: str = None) -> str:
    subdirs = [d for d in glob.glob(os.path.join(results_root, "*")) if os.path.isdir(d)]
    if not subdirs:
        if fallback is None:
            raise FileNotFoundError(f"No subdirectories under {results_root}")
        logger.info(f"[Eval] No subdirs under {results_root}, using fallback {fallback}")
        return fallback
    latest = max(subdirs, key=os.path.getmtime)
    logger.info(f"[Eval] Using latest run dir: {latest}")
    return latest

run_dir = _latest_run_dir(RESULTS_ROOT, fallback=FALLBACK_RUN_DIR)
print(f"[Eval] RUN_DIR = {run_dir}")

RESULTS_HISTORY_PKL = os.path.join(run_dir, RESULTS_PKL_NAME)
PRED_PARQUET_PATH   = os.path.join(run_dir, PRED_PARQUET_NAME)
PRED_CSV_PATH       = os.path.join(run_dir, PRED_CSV_NAME)

print(f"[Eval] RESULTS_HISTORY_PKL: {RESULTS_HISTORY_PKL}")
print(f"[Eval] PRED_PARQUET_PATH:   {PRED_PARQUET_PATH}")
print(f"[Eval] PRED_CSV_PATH:       {PRED_CSV_PATH}")

best_model_candidates = glob.glob(os.path.join(run_dir, "**", "best_model.pth"), recursive=True)
if best_model_candidates:
    BEST_MODEL_PATH = best_model_candidates[0]
    print(f"[Eval] BEST_MODEL_PATH:   {BEST_MODEL_PATH}")
else:
    BEST_MODEL_PATH = None
    print("[Eval] WARNING: No best_model.pth found; metrics will skip model-based fallback.")

# ----------------------------------------------------------
# 2. Load results_summary and predictions DataFrame
# ----------------------------------------------------------
if not os.path.exists(RESULTS_HISTORY_PKL):
    raise FileNotFoundError(f"[Eval] Could not find {RESULTS_HISTORY_PKL}")

with open(RESULTS_HISTORY_PKL, "rb") as f:
    results_summary = pickle.load(f)
print("[Eval] Loaded results_summary from pickle.")

if os.path.exists(PRED_PARQUET_PATH):
    df_pred = pd.read_parquet(PRED_PARQUET_PATH)
    print(f"[Eval] Loaded predictions from Parquet. Shape: {df_pred.shape}")
elif os.path.exists(PRED_CSV_PATH):
    df_pred = pd.read_csv(PRED_CSV_PATH)
    print(f"[Eval] Loaded predictions from CSV. Shape: {df_pred.shape}")
else:
    raise FileNotFoundError(
        f"[Eval] Neither {PRED_PARQUET_PATH} nor {PRED_CSV_PATH} exists."
    )

config = results_summary.get("config_summary", {})
price_col = config.get("PRICE_COL", "sale_price")
log_y_col = results_summary.get("vae_target_y_column", None)

if log_y_col is None:
    raise ValueError("[Eval] 'vae_target_y_column' is None in results_summary – cannot evaluate.")
if log_y_col not in df_pred.columns:
    raise KeyError(f"[Eval] Target column '{log_y_col}' not found in df_pred.")

# ----------------------------------------------------------
# 3. Build VAE inputs and load model/trainer
#     (used for both fallback metrics and latent viz)
# ----------------------------------------------------------
vae_x_cols = results_summary.get("vae_actual_x_features_used", [])
if not vae_x_cols:
    raise ValueError("[Eval] 'vae_actual_x_features_used' is empty – cannot rebuild VAE inputs.")

feature_columns_for_vae = list(dict.fromkeys(vae_x_cols + [log_y_col]))
missing_feats = [c for c in feature_columns_for_vae if c not in df_pred.columns]
if missing_feats:
    print(f"[Eval] WARNING: some VAE feature columns missing in df_pred: {missing_feats}")

(X_filled_np, X_mask_np,
 y_filled_np, y_mask_np,
 feature_names_x_out, _, _) = prepare_vae_input(
    df=df_pred.copy(),
    feature_columns=feature_columns_for_vae,
    target_col=log_y_col,
    scaler_type="standard"
)

print(f"[Eval] VAE input shapes: X={X_filled_np.shape}, Y={y_filled_np.shape}")

artifacts_summary = results_summary.get("orchestrator_artifacts_summary", {})
latent_dim_suggestion = artifacts_summary.get("latent_dim_suggestion", 3)
latent_dim_override = max(3, int(latent_dim_suggestion))

vae_model = create_vae_from_artifacts(
    artifacts=artifacts_summary,
    feature_order=feature_names_x_out + [log_y_col],
    target_col_name=log_y_col,
    alpha_price_loss=config.get("VAE_ALPHA_PRICE_LOSS", 10.0),
    latent_dim_override=latent_dim_override,
    device=DEVICE
)

if BEST_MODEL_PATH is not None and os.path.exists(BEST_MODEL_PATH):
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    vae_model.load_state_dict(state_dict)
    print("[Eval] Loaded model_state_dict into VAE model from best_model.pth")
else:
    print("[Eval] WARNING: Using fresh-initialized MIWAE weights (no best_model.pth found).")

vae_trainer = VAETrainer(
    model=vae_model,
    learning_rate=config.get("VAE_LEARNING_RATE", 3e-4),
    device=DEVICE,
    verbose=False,
    reconstruction_loss_type_x=config.get("VAE_RECON_LOSS_X", "mse"),
    price_loss_type_y=config.get("VAE_PRICE_LOSS_Y", "gaussian_nll"),
    alpha_price_loss=config.get("VAE_ALPHA_PRICE_LOSS", 10.0),
    kld_weight=1.0,
)

# ----------------------------------------------------------
# 4. Posterior predictive metrics
#     (use cv_* columns if present; otherwise fallback to best_model)
#     + store residuals for diagnostics
# ----------------------------------------------------------
pred_price_col_cv = f"cv_predicted_{price_col}"
pred_std_col_cv   = f"cv_pred_uncertainty_{log_y_col}_std"
pred_flag_col_cv  = "cv_prediction_available"

have_cv = all(c in df_pred.columns for c in [pred_price_col_cv, pred_std_col_cv, pred_flag_col_cv])

# For later residual diagnostics (log space)
y_true_log_eval = None
mu_log_eval = None

def _report_metrics(y_true_log, y_true, mu_log, var_log, label_prefix=""):
    var_log = np.clip(var_log, 1e-9, np.inf)

    # Gaussian log-likelihood in log-price space
    log_pp = -0.5 * (np.log(2 * math.pi * var_log) + ((y_true_log - mu_log) ** 2) / var_log)
    avg_log_pp = float(np.mean(log_pp))
    avg_nll_log = -avg_log_pp

    # basic errors in log space
    resid_log = y_true_log - mu_log
    mse_log = float(np.mean(resid_log ** 2))
    rmse_log = math.sqrt(mse_log)
    mae_log = float(np.mean(np.abs(resid_log)))

    # robust log-space metrics for the writeup
    abs_resid_log = np.abs(resid_log)
    median_abs_log = float(np.median(abs_resid_log))
    p95_abs_log    = float(np.percentile(abs_resid_log, 95))

    # errors in level space
    mse_price = float(np.mean((y_true - np.exp(mu_log)) ** 2))
    rmse_price = math.sqrt(mse_price)
    mae_price = float(np.mean(np.abs(y_true - np.exp(mu_log))))

    positive_mask = y_true > 0
    if positive_mask.sum() > 0:
        mape_price = float(
            np.mean(
                np.abs(
                    (y_true[positive_mask] - np.exp(mu_log[positive_mask]))
                    / y_true[positive_mask]
                )
            )
        )
    else:
        mape_price = float("nan")

    tag = f" ({label_prefix})" if label_prefix else ""
    print(f"\n=== Posterior Predictive Metrics{tag} ===")
    print(f"Average log posterior predictive (log p(y_log | x)): {avg_log_pp:.4f}")
    print(f"Average NLL (log-price space):                     {avg_nll_log:.4f}")
    print(f"RMSE (log-price):                                  {rmse_log:.4f}")
    print(f"MAE  (log-price):                                  {mae_log:.4f}")
    print(f"RMSE (price):                                      {rmse_price:,.4f}")
    print(f"MAE  (price):                                      {mae_price:,.4f}")
    print(f"MAPE (price, y_true > 0):                          {mape_price * 100:,.2f}%")
    print(f"Median |y - ŷ| (log):                              {median_abs_log:.4f}")
    print(f"95th pct |y - ŷ| (log):                            {p95_abs_log:.4f}")

if have_cv:
    print("\n[Eval] Found cv_* prediction columns; using CV-stitched predictions for held-out metrics.")
    mask_pred = df_pred[pred_flag_col_cv].astype(bool).values
    df_eval = df_pred.loc[mask_pred].copy()
    print(f"[Eval] Rows with CV predictions: {df_eval.shape[0]} / {df_pred.shape[0]}")

    y_true_log = df_eval[log_y_col].astype(float).values
    y_true     = df_eval[price_col].astype(float).values
    y_pred     = df_eval[pred_price_col_cv].astype(float).values
    sigma_log  = df_eval[pred_std_col_cv].astype(float).values

    mu_log = np.log(np.clip(y_pred, 1e-12, np.inf))
    var_log = np.square(sigma_log)

    _report_metrics(y_true_log, y_true, mu_log, var_log, label_prefix="CV stitched")

    # store for residual diagnostics
    y_true_log_eval = y_true_log
    mu_log_eval = mu_log

    # optional metrics on "uncertainty OK" subset
    unc_ok_col = "prediction_uncertainty_ok"
    if unc_ok_col in df_eval.columns:
        mask_unc_ok = df_eval[unc_ok_col].astype(bool).values
        if mask_unc_ok.sum() > 0:
            print(f"\n[Eval] Metrics on subset with {unc_ok_col} == True")
            y_true_log_ok = y_true_log[mask_unc_ok]
            y_true_ok     = y_true[mask_unc_ok]
            mu_log_ok     = mu_log[mask_unc_ok]
            var_log_ok    = var_log[mask_unc_ok]
            _report_metrics(y_true_log_ok, y_true_ok, mu_log_ok, var_log_ok,
                            label_prefix="CV stitched, unc_ok")

else:
    print("\n[Eval] cv_* columns not found; falling back to direct predictions from best_model on a random hold-out split.")
    if BEST_MODEL_PATH is None:
        raise RuntimeError("[Eval] No cv_* columns and no best_model.pth; cannot compute predictive metrics.")

    # mask for rows where the target is observed
    y_mask = y_mask_np.squeeze()
    obs_idx = np.where(y_mask > 0.0)[0]
    if obs_idx.size == 0:
        raise RuntimeError("[Eval] No observed targets according to VAE mask; cannot evaluate.")

    y_true_log_all = df_pred[log_y_col].astype(float).to_numpy()[obs_idx]
    y_true_all     = df_pred[price_col].astype(float).to_numpy()[obs_idx]
    X_obs = X_filled_np[obs_idx]

    N_obs = X_obs.shape[0]
    rng = np.random.default_rng(42)
    perm = rng.permutation(N_obs)
    split = int(0.8 * N_obs)
    test_rel_idx = perm[split:]
    X_test = X_obs[test_rel_idx]
    y_true_log = y_true_log_all[test_rel_idx]
    y_true     = y_true_all[test_rel_idx]

    X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)
    log_mu, log_var = vae_trainer.predict_price_and_uncertainty(X_test_tensor, batch_size=1024)
    log_mu = np.asarray(log_mu).reshape(-1)
    log_var = np.asarray(log_var).reshape(-1)
    var_log = np.exp(log_var)

    print(f"[Eval] Evaluating on random hold-out: {len(test_rel_idx)} / {N_obs} observed rows.")
    _report_metrics(y_true_log, y_true, log_mu, var_log, label_prefix="best_model, random hold-out")

    # store for residual diagnostics
    y_true_log_eval = y_true_log
    mu_log_eval = log_mu

# ===========================================================================
# 5. Convergence diagnostics (train/val loss vs epoch)
# ===========================================================================
def _extract_fold_histories(summary_dict):
    """Try to recover a list of history dicts from several possible layouts."""
    fold_histories_local = []
    vae_hist = summary_dict.get("vae_training_history", None)

    # Case A: multi-fold
    if isinstance(vae_hist, dict) and isinstance(vae_hist.get("fold_histories"), list):
        fold_histories_local = vae_hist["fold_histories"]
        print(f"[Eval] Found fold_histories (n_folds={len(fold_histories_local)}).")
        return fold_histories_local

    # Case B: nested single
    if isinstance(vae_hist, dict) and isinstance(vae_hist.get("history"), dict):
        fold_histories_local = [vae_hist["history"]]
        print("[Eval] Found single history; treating as one fold.")
        return fold_histories_local

    # Case C: flat dict with train_/val_ keys
    if isinstance(vae_hist, dict):
        has_train_val = any(k.startswith("train_") or k.startswith("val_") for k in vae_hist.keys())
        if has_train_val:
            fold_histories_local = [vae_hist]
            print("[Eval] Flat history dict; treating as one fold.")
            return fold_histories_local

    # Case D: top-level history
    top_hist = summary_dict.get("history", None)
    if isinstance(top_hist, dict):
        fold_histories_local = [top_hist]
        print("[Eval] Found top-level history; treating as one fold.")
        return fold_histories_local

    return []

fold_histories = _extract_fold_histories(results_summary)

if not fold_histories:
    print("\n[Eval] No usable training histories found; skipping convergence plots.")
else:
    n_folds = len(fold_histories)
    print(f"\n[Eval] Plotting convergence diagnostics (n_folds={n_folds}).")

    fig, ax = plt.subplots(figsize=(8, 5))
    
    all_train_losses = []
    all_val_losses = []
    max_len = 0

    for i, hist in enumerate(fold_histories):
        train_key = next((k for k in ["train_total_loss", "train_loss", "train_elbo"] if k in hist), None)
        val_key   = next((k for k in ["val_total_loss", "val_loss", "val_elbo"] if k in hist), None)
        
        if train_key and val_key:
            t_loss = np.array(hist[train_key], dtype=float)
            v_loss = np.array(hist[val_key], dtype=float)
            all_train_losses.append(t_loss)
            all_val_losses.append(v_loss)
            max_len = max(max_len, len(t_loss))
            
            # Plot individual folds faintly
            epochs = np.arange(1, len(t_loss) + 1)
            ax.plot(epochs, t_loss, alpha=0.2, color='#1f77b4', linewidth=0.8)
            ax.plot(epochs, v_loss, alpha=0.2, color='#ff7f0e', linestyle="--", linewidth=0.8)

    # Compute and plot averages (or single values)
    if all_train_losses:
        train_matrix = np.full((len(all_train_losses), max_len), np.nan)
        val_matrix = np.full((len(all_val_losses), max_len), np.nan)
        
        for idx, (t, v) in enumerate(zip(all_train_losses, all_val_losses)):
            train_matrix[idx, :len(t)] = t
            val_matrix[idx, :len(v)] = v
            
        mean_train = np.nanmean(train_matrix, axis=0)
        mean_val = np.nanmean(val_matrix, axis=0)
        ep = np.arange(1, max_len + 1)
        
        # Legend labels conditional on number of folds
        if n_folds > 1:
            train_label, val_label = "Train (mean)", "Val (mean)"
        else:
            train_label, val_label = "Train", "Validation"
        
        ax.plot(ep, mean_train, color='#1f77b4', linewidth=2.5, label=train_label)
        ax.plot(ep, mean_val, color='#ff7f0e', linestyle="--", linewidth=2.5, label=val_label)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Total Loss", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    
    # Title - centered on axes, not figure
    ax.set_title("SemiSupMIWAE Convergence", fontsize=14, fontweight='bold', pad=10)
    
    # Footnote explaining loss function
    ax.text(0.5, -0.12, "Total Loss = Reconstruction Loss + β·KL Divergence",
            transform=ax.transAxes, fontsize=8, color='gray', ha='center')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# ===========================================================================
# 6. Residual diagnostics in log-price space
# ===========================================================================
if y_true_log_eval is not None and mu_log_eval is not None:
    print("\n[Eval] Generating residual diagnostics.")

    resid_log = y_true_log_eval - mu_log_eval
    
    # Fit parameters (used across all residual plots)
    if HAVE_SCIPY:
        mu_fit, std_fit = stats.norm.fit(resid_log)
        df_fit, loc_fit, scale_fit = stats.t.fit(resid_log)
        x_max = max(abs(resid_log.min()), abs(resid_log.max())) + 0.5
        x_grid = np.linspace(-x_max, x_max, 500)
    
    # --- VERSION 1: Simple histogram only ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resid_log, bins=50, density=True, alpha=0.8, color='#2ecc71', 
            edgecolor='white', linewidth=0.3)
    ax.set_xlabel("Residual", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(-x_max, x_max)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    fig.suptitle("Residual Distribution", fontsize=14, fontweight='bold', y=0.98)
    ax.set_title(r"$r = \log(y_{\mathrm{true}}) - \log(\hat{y})$", fontsize=9, color='gray', pad=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # --- VERSION 2: Histogram + Normal reference only ---
    if HAVE_SCIPY:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(resid_log, bins=50, density=True, alpha=0.8, color='#2ecc71', 
                edgecolor='white', linewidth=0.3, label='Observed', zorder=3)
        pdf_norm = stats.norm.pdf(x_grid, loc=mu_fit, scale=std_fit)
        ax.plot(x_grid, pdf_norm, color='gray', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='Normal (MLE fit)', zorder=1)
        ax.set_xlabel("Residual", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_xlim(-x_max, x_max)
        ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        fig.suptitle("Residuals vs Normal", fontsize=14, fontweight='bold', y=0.98)
        ax.set_title("Log-transformed prices", fontsize=9, color='gray', pad=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    # --- VERSION 3: Full with Student-t overlays ---
    
    # --- 6a. Histogram with Student-t overlays ---
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Main histogram - prominent color
    ax.hist(resid_log, bins=50, density=True, alpha=0.8, color='#2ecc71', 
            edgecolor='white', linewidth=0.3, label='Observed residuals', zorder=3)
    
    if HAVE_SCIPY:
        # Use already-fitted parameters from above (df_fit, loc_fit, scale_fit)
        
        # Student-t references: use MULTIPLICATIVE ratios around fitted df
        # E.g., if fitted df=5, show df=2.5 (heavier, ν/2) and df=10 (lighter, ν*2)
        df_lower = max(2.1, df_fit / 2)  # heavier tails (half the df)
        df_upper = df_fit * 2  # lighter tails (double the df)
        
        for nu_ref, style, lbl in [(df_lower, ':', f'ν={df_lower:.1f} (heavier)'), 
                                    (df_upper, '-.', f'ν={df_upper:.1f} (lighter)')]:
            # Match variance: scale = std / sqrt(nu/(nu-2))
            if nu_ref > 2:
                scale_ref = std_fit / np.sqrt(nu_ref / (nu_ref - 2))
            else:
                scale_ref = std_fit * 0.5
            pdf_t_ref = stats.t.pdf(x_grid, df=nu_ref, loc=mu_fit, scale=scale_ref)
            ax.plot(x_grid, pdf_t_ref, color='gray', linestyle=style, linewidth=1.2, 
                    alpha=0.6, label=f'Student-t {lbl}', zorder=1)
        
        # Best-fit Student-t (prominent)
        pdf_t_best = stats.t.pdf(x_grid, df=df_fit, loc=loc_fit, scale=scale_fit)
        ax.plot(x_grid, pdf_t_best, color='#e74c3c', linestyle='-', linewidth=2, 
                alpha=0.9, label=f'Student-t fit (ν={df_fit:.1f})', zorder=2)

    ax.set_xlabel("Residual", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    
    # Center x-axis on 0
    ax.set_xlim(-x_max, x_max)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)  # subtle zero line
    
    # Title + subtitle with formula
    fig.suptitle("Residuals vs Heavy-Tailed References", fontsize=14, fontweight='bold', y=0.98)
    ax.set_title(r"$r = \log(y_{\mathrm{true}}) - \log(\hat{y})$  ·  Log-transformed prices", 
                 fontsize=9, color='gray', pad=3)
    
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # --- 6b. QQ-plot vs Student-t reference ---
    if HAVE_SCIPY:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Use Student-t as reference distribution (not Normal)
        # Generate theoretical Student-t quantiles
        n_samples = len(resid_log)
        
        # Theoretical quantiles from fitted Student-t
        p = (np.arange(1, n_samples + 1) - 0.5) / n_samples
        theoretical_quantiles = stats.t.ppf(p, df=df_fit, loc=loc_fit, scale=scale_fit)
        
        # Ordered sample quantiles
        sample_quantiles = np.sort(resid_log)
        
        # Fit line (should be close to y=x for good fit)
        slope_t, intercept_t = np.polyfit(theoretical_quantiles, sample_quantiles, 1)
        r_t = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
        
        # Bootstrap confidence bands for Student-t
        n_boot = 200
        rng = np.random.default_rng(42)
        
        sim_t = np.sort(stats.t.rvs(df=df_fit, loc=loc_fit, scale=scale_fit, 
                                     size=(n_boot, n_samples), random_state=rng), axis=1)
        lower_t = np.percentile(sim_t, 2.5, axis=0)
        upper_t = np.percentile(sim_t, 97.5, axis=0)
        
        # Plot confidence band (solid fill, prominent)
        ax.fill_between(theoretical_quantiles, lower_t, upper_t, 
                        color='#e74c3c', alpha=0.2, 
                        label=f'95% CI (Student-t ν={df_fit:.1f})', zorder=1)
        
        # Reference line: y = x (perfect fit)
        ref_line = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
        ax.plot(ref_line, ref_line, color='#e74c3c', linestyle='--', 
                linewidth=2, alpha=0.8, label='Perfect fit (y=x)', zorder=2)
        
        # Data points - layered transparency
        ax.scatter(theoretical_quantiles, sample_quantiles, c='#2ecc71', s=12, 
                   alpha=0.4, edgecolors='none', zorder=3)
        step = max(1, n_samples // 100)
        ax.scatter(theoretical_quantiles[::step], sample_quantiles[::step], 
                   c='#2ecc71', s=20, alpha=0.9, 
                   edgecolors='#1a7a3e', linewidths=0.5, zorder=4)
        
        ax.set_xlabel(f"Theoretical Quantiles (Student-t, ν={df_fit:.1f})", fontsize=11)
        ax.set_ylabel("Ordered Residuals", fontsize=11)
        
        # Title
        fig.suptitle("Q-Q Plot", fontsize=14, fontweight='bold', y=0.98)
        ax.set_title("Residuals vs Student-t reference", fontsize=9, color='gray', pad=3)
        
        # Annotation for fit quality
        ax.text(0.05, 0.95, f"Slope: {slope_t:.3f}\n$R^2$: {r_t**2:.3f}", 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        all_vals = np.concatenate([theoretical_quantiles, sample_quantiles])
        lim = max(abs(all_vals.min()), abs(all_vals.max())) * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        print("[Eval] Skipping QQ-plot (scipy not available).")
else:
    print("\n[Eval] No stored residuals; skipping residual diagnostics.")

# ===========================================================================
# 7. Latent-space visualization
# ===========================================================================
print("\n[Eval] Generating latent space visualizations.")

X_all_np = X_filled_np.astype(np.float32, copy=False)
try:
    mu_z, logvar_z = vae_trainer.get_latent_representation(X_all_np, batch_size=1024)
    mu_z = np.asarray(mu_z)
except AttributeError:
    if hasattr(vae_model, "encode"):
        mu_list = []
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(X_all_np), batch_size):
                xb = torch.from_numpy(X_all_np[i:i+batch_size]).to(DEVICE)
                res = vae_model.encode(xb)
                if isinstance(res, tuple):
                    mu_list.append(res[0].cpu().numpy())
                else:
                    mu_list.append(res.loc.cpu().numpy())
        mu_z = np.concatenate(mu_list, axis=0)
    else:
        print("[Eval] Warning: Could not extract latent means.")
        mu_z = np.zeros((len(X_all_np), 2))

if mu_z.ndim == 2 and mu_z.shape[1] >= 2:
    
    # Compute per-dimension statistics to decide on axis handling
    z1_std, z2_std = np.std(mu_z[:, 0]), np.std(mu_z[:, 1])
    z1_range = np.ptp(mu_z[:, 0])  # peak-to-peak
    z2_range = np.ptp(mu_z[:, 1])
    
    print(f"[Eval] Latent dimension stats: z1_std={z1_std:.3f}, z2_std={z2_std:.3f}")
    print(f"[Eval] Latent dimension ranges: z1_range={z1_range:.3f}, z2_range={z2_range:.3f}")
    
    # Decision: use equal axes only if variances are within 2x of each other
    # Otherwise, let each axis have its own scale to honestly represent structure
    variance_ratio = max(z1_std, z2_std) / min(z1_std, z2_std) if min(z1_std, z2_std) > 0 else 1
    use_equal_axes = variance_ratio < 2.0
    
    if use_equal_axes:
        z_abs_max = max(np.abs(mu_z[:, 0]).max(), np.abs(mu_z[:, 1]).max()) * 1.05
        z_lim = (-z_abs_max, z_abs_max)
        print(f"[Eval] Using equal axes (variance ratio={variance_ratio:.2f} < 2.0)")
    else:
        z_lim = None  # Will use per-axis limits
        print(f"[Eval] Using per-axis limits (variance ratio={variance_ratio:.2f} >= 2.0)")
    
    # Helper for KDE density contours
    def add_density_contours(x, y, ax, levels=5, color='white', alpha=0.6):
        """Add KDE density contour lines to a plot."""
        if not HAVE_SCIPY:
            return
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
            ax.contour(xi, yi, zi, levels=levels, colors=color, alpha=alpha, linewidths=0.8)
        except Exception as e:
            print(f"[Eval] Contour error: {e}")
    
    # --- 7a. Latent colored by SALE PRICE (continuous log scale) ---
    log_price_all = df_pred[log_y_col].astype(float).values
    
    # Filter: require price >= $100,000 (log >= 11.51) to exclude anomalies
    MIN_PRICE_LOG = np.log(100_000)  # ~11.51
    mask_valid_price = np.isfinite(log_price_all) & (log_price_all >= MIN_PRICE_LOG)
    
    print(f"[Eval] Price filter: {mask_valid_price.sum()} / {np.isfinite(log_price_all).sum()} " +
          f"properties with sale price >= $100K")
    
    if mask_valid_price.sum() > 100:
        valid_log_price = log_price_all[mask_valid_price]
        valid_z_price = mu_z[mask_valid_price]
        
        # --- VERSION 1: Scatter + Contour ---
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Add density contours first (behind scatter)
        add_density_contours(valid_z_price[:, 0], valid_z_price[:, 1], ax, 
                            levels=6, color='white', alpha=0.7)
        
        # Scatter with continuous log-price coloring
        sc = ax.scatter(valid_z_price[:, 0], valid_z_price[:, 1], 
                        c=valid_log_price, s=8, alpha=0.5, cmap="viridis",
                        edgecolors='none', zorder=2)
        
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Sale Price", fontsize=10)
        
        # Log-scale tick labels - extended to billions
        vmin, vmax = valid_log_price.min(), valid_log_price.max()
        log_ticks = [np.log(100_000), np.log(250_000), np.log(500_000), 
                     np.log(1_000_000), np.log(2_500_000), np.log(5_000_000),
                     np.log(10_000_000), np.log(25_000_000), np.log(50_000_000),
                     np.log(100_000_000), np.log(250_000_000), np.log(500_000_000),
                     np.log(1_000_000_000), np.log(2_500_000_000), np.log(5_000_000_000)]
        log_labels = ['$100K', '$250K', '$500K', '$1M', '$2.5M', '$5M', '$10M', '$25M', '$50M',
                      '$100M', '$250M', '$500M', '$1B', '$2.5B', '$5B']
        
        # Filter to range and always include min/max
        valid_ticks = [(t, l) for t, l in zip(log_ticks, log_labels) if vmin <= t <= vmax]
        
        # Format price with B for billions
        def format_price(val):
            price = np.exp(val)
            if price >= 1_000_000_000:
                return f"${price/1_000_000_000:.1f}B"
            elif price >= 1_000_000:
                return f"${price/1_000_000:.1f}M"
            else:
                return f"${price/1_000:.0f}K"
        
        final_ticks = [vmin] + [t for t, l in valid_ticks] + [vmax]
        final_labels = [format_price(vmin)] + [l for t, l in valid_ticks] + [format_price(vmax)]
        # Remove duplicates
        seen = set()
        unique_ticks, unique_labels = [], []
        for t, l in zip(final_ticks, final_labels):
            if round(t, 2) not in seen:
                seen.add(round(t, 2))
                unique_ticks.append(t)
                unique_labels.append(l)
        cbar.set_ticks(unique_ticks)
        cbar.set_ticklabels(unique_labels)
        
        ax.set_xlabel("Latent Dimension 1", fontsize=11)
        ax.set_ylabel("Latent Dimension 2", fontsize=11)
        
        if use_equal_axes:
            ax.set_xlim(z_lim)
            ax.set_ylim(z_lim)
            ax.set_aspect('equal', adjustable='box')
        
        # Title centered on axes
        ax.set_title("Latent Space by Price\nScatter with density contours", 
                     fontsize=12, fontweight='bold', pad=8)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # --- VERSION 2: Hexbin by price (mean log-price per bin) ---
        fig, ax = plt.subplots(figsize=(7, 6))
        
        hb = ax.hexbin(valid_z_price[:, 0], valid_z_price[:, 1], 
                       C=valid_log_price, reduce_C_function=np.mean,
                       gridsize=40, cmap='viridis', mincnt=1, linewidths=0.2)
        
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label("Mean Sale Price", fontsize=10)
        
        # Same log-scale labels
        if valid_ticks:
            cbar.set_ticks([t for t, l in valid_ticks])
            cbar.set_ticklabels([l for t, l in valid_ticks])
        
        ax.set_xlabel("Latent Dimension 1", fontsize=11)
        ax.set_ylabel("Latent Dimension 2", fontsize=11)
        
        if use_equal_axes:
            ax.set_xlim(z_lim)
            ax.set_ylim(z_lim)
            ax.set_aspect('equal', adjustable='box')
        
        fig.suptitle("Latent Space by Price", fontsize=14, fontweight='bold', y=0.98)
        ax.set_title("Hexbin showing mean price per region", fontsize=9, color='gray', pad=3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    # --- 7a-bis. Hexbin density plot for latent space (count only) ---
    if mask_valid_price.sum() > 100:
        fig, ax = plt.subplots(figsize=(7, 6))
        
        hb = ax.hexbin(valid_z_price[:, 0], valid_z_price[:, 1], 
                       gridsize=40, cmap='Blues', mincnt=1, linewidths=0.2)
        
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label("Count", fontsize=10)
        
        ax.set_xlabel("Latent Dimension 1", fontsize=11)
        ax.set_ylabel("Latent Dimension 2", fontsize=11)
        
        if use_equal_axes:
            ax.set_xlim(z_lim)
            ax.set_ylim(z_lim)
            ax.set_aspect('equal', adjustable='box')
        
        fig.suptitle("Latent Space Density", fontsize=14, fontweight='bold', y=0.98)
        ax.set_title("Hexbin showing point concentration", fontsize=9, color='gray', pad=3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # --- 7b. Latent colored by SIZE (Square Footage) ---
    size_candidates = ["gross_sqft", "land_sqft", "gross_square_feet", "land_area", "sqft", "area"]
    size_col = next((c for c in size_candidates if c in df_pred.columns), None)
    
    if size_col:
        print(f"[Eval] Using '{size_col}' for Size-based Latent Viz.")
        size_vals = df_pred[size_col].astype(float).values
        mask_valid_size = np.isfinite(size_vals) & (size_vals > 0)
        
        if mask_valid_size.sum() > 100:
            log_size = np.log10(size_vals[mask_valid_size])
            valid_z_size = mu_z[mask_valid_size]
            
            # --- VERSION 1: Scatter + Contour ---
            fig, ax = plt.subplots(figsize=(7, 6))
            
            add_density_contours(valid_z_size[:, 0], valid_z_size[:, 1], ax,
                                levels=6, color='white', alpha=0.7)
            
            sc = ax.scatter(valid_z_size[:, 0], valid_z_size[:, 1], c=log_size, s=8, alpha=0.5, 
                           cmap="magma", edgecolors='none', zorder=2)
            
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Square Footage", fontsize=10)
            # Log-scale ticks for sqft
            sqft_ticks = [2, 2.5, 3, 3.5, 4, 4.5, 5]  # log10 values
            sqft_labels = ['100', '300', '1K', '3K', '10K', '30K', '100K']
            vmin, vmax = log_size.min(), log_size.max()
            valid_sqft = [(t, l) for t, l in zip(sqft_ticks, sqft_labels) if vmin <= t <= vmax]
            if valid_sqft:
                cbar.set_ticks([t for t, l in valid_sqft])
                cbar.set_ticklabels([l for t, l in valid_sqft])
            
            ax.set_xlabel("Latent Dimension 1", fontsize=11)
            ax.set_ylabel("Latent Dimension 2", fontsize=11)
            
            if use_equal_axes:
                ax.set_xlim(z_lim)
                ax.set_ylim(z_lim)
                ax.set_aspect('equal', adjustable='box')
            
            fig.suptitle("Latent Space by Size", fontsize=14, fontweight='bold', y=0.98)
            ax.set_title(f"Scatter with density contours · {size_col}", fontsize=9, color='gray', pad=3)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
            
            # --- VERSION 2: Hexbin by size ---
            fig, ax = plt.subplots(figsize=(7, 6))
            
            hb = ax.hexbin(valid_z_size[:, 0], valid_z_size[:, 1], 
                           C=log_size, reduce_C_function=np.mean,
                           gridsize=40, cmap='magma', mincnt=1, linewidths=0.2)
            
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label("Mean Sq Ft", fontsize=10)
            if valid_sqft:
                cbar.set_ticks([t for t, l in valid_sqft])
                cbar.set_ticklabels([l for t, l in valid_sqft])
            
            ax.set_xlabel("Latent Dimension 1", fontsize=11)
            ax.set_ylabel("Latent Dimension 2", fontsize=11)
            
            if use_equal_axes:
                ax.set_xlim(z_lim)
                ax.set_ylim(z_lim)
                ax.set_aspect('equal', adjustable='box')
            
            fig.suptitle("Latent Space by Size", fontsize=14, fontweight='bold', y=0.98)
            ax.set_title("Hexbin showing mean size per region", fontsize=9, color='gray', pad=3)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        else:
            print("[Eval] Not enough valid size data for plot.")
    else:
        print("[Eval] No size/sqft column found; skipping Size latent plot.")

    # --- 7c. Latent colored by Building Class (AGGREGATED) ---
    bldg_class_candidates = ["bldg_class", "building_class", "bldg_class_group", "bldgclass"]
    bldg_class_col = next((c for c in bldg_class_candidates if c in df_pred.columns), None)
    
    if bldg_class_col:
        print(f"[Eval] Using '{bldg_class_col}' for Building Class Viz (aggregated to first letter).")
        
        # Aggregate to first letter (major category)
        bldg_series = df_pred[bldg_class_col].astype(str).str[0].str.upper()
        
        # NYC Building Class codes - consumer-facing order (like StreetEasy/Zillow)
        # Order: Condos/Co-ops -> Multi-family Rental -> Houses -> Commercial -> Industrial -> Other
        # This matches how typical NYC real estate listings organize property types
        class_labels_ordered = [
            # Most common residential (what buyers/renters search for)
            ('R', 'Condominiums'),
            ('D', 'Elevator Apartments'),
            ('C', 'Walk-up Apartments'),
            ('S', 'Mixed Residential'),
            # Houses
            ('A', '1-2 Family Houses'),
            ('B', '2 Family Houses'),
            # Commercial/Mixed-use
            ('K', 'Retail/Stores'),
            ('O', 'Office Buildings'),
            ('H', 'Hotels'),
            ('L', 'Lofts'),
            # Industrial
            ('E', 'Warehouses'),
            ('F', 'Factories'),
            ('G', 'Garages'),
            # Special Purpose
            ('I', 'Healthcare'),
            ('J', 'Entertainment'),
            ('M', 'Religious'),
            ('N', 'Nursing/Asylums'),
            ('P', 'Recreation (Indoor)'),
            ('Q', 'Recreation (Outdoor)'),
            ('T', 'Transportation'),
            ('W', 'Educational'),
            # Other
            ('U', 'Utility'),
            ('V', 'Vacant Land'),
            ('Y', 'Government'),
            ('Z', 'Miscellaneous'),
        ]
        class_labels = {k: v for k, v in class_labels_ordered}
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use colormap with colors assigned in the PREDEFINED ORDER
        # This ensures legend order matches color order
        present_classes = [letter for letter, _ in class_labels_ordered if letter in bldg_series.values]
        n_present = len(present_classes)
        cmap = plt.cm.get_cmap('tab20', max(n_present, 1))
        
        # Plot in predefined order (not data occurrence order)
        for color_idx, letter in enumerate(present_classes):
            mask = (bldg_series == letter).values
            if not np.any(mask): 
                continue
            lbl = class_labels.get(letter, letter)
            ax.scatter(mu_z[mask, 0], mu_z[mask, 1], s=8, alpha=0.5, 
                      color=cmap(color_idx), label=lbl, edgecolors='none')
        
        ax.set_xlabel("Latent Dimension 1", fontsize=11)
        ax.set_ylabel("Latent Dimension 2", fontsize=11)
        
        if use_equal_axes:
            ax.set_xlim(z_lim)
            ax.set_ylim(z_lim)
            ax.set_aspect('equal', adjustable='box')
        
        ax.set_title("Latent Space by Building Class\nNYC DOF Classification", 
                     fontsize=12, fontweight='bold', pad=8)
        
        # Legend outside if many classes
        if n_present <= 8:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
        else:
            ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                     framealpha=0.9, ncol=1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.95] if len(uniques) > 8 else [0, 0, 1, 0.95])
        plt.show()

    # --- 7d. Marginal histograms ---
    n_dims = min(mu_z.shape[1], 3)
    fig, axes = plt.subplots(1, n_dims, figsize=(4*n_dims, 3.5))
    if n_dims == 1:
        axes = [axes]
    
    for d, ax in enumerate(axes):
        ax.hist(mu_z[:, d], bins=40, alpha=0.8, color='#3498db', 
                edgecolor='white', linewidth=0.3, density=True)
        ax.set_xlabel(f"Latent Dim {d+1}", fontsize=10)
        ax.set_ylabel("Density" if d == 0 else "", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle("Latent Marginal Distributions", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

print("[Eval] All visualizations completed.")
