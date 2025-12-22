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

def save_figure(filename):
    """Helper to save the current figure to the run directory."""
    try:
        path = os.path.join(run_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=150)
        print(f"[Eval] Saved plot to: {path}")
    except Exception as e:
        print(f"[Eval] Failed to save plot {filename}: {e}")

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
    var_log_eval = var_log
    # Store indices to fetch metadata (year, building class) for segment analysis
    eval_indices = df_eval.index.values

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

    _report_metrics(y_true_log, y_true, log_mu, var_log, label_prefix="best_model, random hold-out")

    # store for residual diagnostics
    y_true_log_eval = y_true_log
    mu_log_eval = log_mu
    var_log_eval = var_log
    # Convert relative indices back to original dataframe indices
    # obs_idx indexes into X_filled_np (which matches df_pred)
    # test_rel_idx indexes into obs_idx
    eval_indices = df_pred.index.values[obs_idx][test_rel_idx]

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
    
    save_figure("convergence_loss.png")
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
    ax.set_xlim(-10, 10)  # Fixed x-axis limits
    ax.set_ylim(0, 2)  # Fixed y-axis limits
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    fig.suptitle("Residual Distribution", fontsize=14, fontweight='bold', y=0.98)
    ax.set_title(r"$r = \log(y_{\mathrm{true}}) - \log(\hat{y})$", fontsize=9, color='gray', pad=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure("residuals_hist_simple.png")
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
        ax.set_xlim(-10, 10)  # Fixed x-axis limits
        ax.set_ylim(0, 2)  # Fixed y-axis limits
        ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        fig.suptitle("Residuals vs Normal", fontsize=14, fontweight='bold', y=0.98)
        ax.set_title("Log-transformed prices", fontsize=9, color='gray', pad=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure("residuals_hist_normal.png")
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
        # E.g., if fitted df=0.7, show df=0.35 (heavier, ν/2) and df=1.4 (lighter, ν*2)
        # Reference lines: wider range to show contrast (10x factor)
        # Student-t is valid for any df > 0
        df_lower = max(0.1, df_fit / 10.0)
        df_upper = df_fit * 10.0
        
        # Reference lines: same gray color, different dash patterns
        # Lower ν (heavier tails) = sparser dashes, higher ν (lighter) = denser dashes
        for nu_ref, style in [(df_lower, (0, (5, 10))),   # sparse dash = heavier
                              (df_upper, (0, (3, 3)))]:   # dense dash = lighter
            # Use same location and scale as the fit to compare shape (tail) only
            # Matching variance is undefined for nu <= 2
            pdf_t_ref = stats.t.pdf(x_grid, df=nu_ref, loc=loc_fit, scale=scale_fit)
            ax.plot(x_grid, pdf_t_ref, color='gray', linestyle=style, linewidth=1.2, 
                    alpha=0.7, label=f'ν={nu_ref:.1f}', zorder=1)
        
        # Best-fit Student-t: same gray color but SOLID line (not dashed)
        pdf_t_best = stats.t.pdf(x_grid, df=df_fit, loc=loc_fit, scale=scale_fit)
        ax.plot(x_grid, pdf_t_best, color='gray', linestyle='-', linewidth=2, 
                alpha=0.9, label=f'Student-t fit (ν={df_fit:.1f})', zorder=2)

    ax.set_xlabel("Residual", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    
    # Fixed axis limits for residual plots
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 2)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)  # subtle zero line
    
    # Title + subtitle with formula
    fig.suptitle("Residuals vs Heavy-Tailed References", fontsize=14, fontweight='bold', y=0.98)
    ax.set_title(r"$r = \log(y_{\mathrm{true}}) - \log(\hat{y})$  ·  Log-transformed prices", 
                 fontsize=9, color='gray', pad=3)
    
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure("residuals_hist_studentt.png")
    plt.show()

    # --- 6b-ORIG. ORIGINAL Simple QQ-plot vs Normal (for comparison) ---
    if HAVE_SCIPY:
        fig, ax = plt.subplots(figsize=(5, 5))
        stats.probplot(resid_log, dist="norm", plot=ax)
        ax.set_title("QQ-plot: Residuals vs Normal\n(Original Simple Version)", 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel("Theoretical Quantiles (Normal)")
        ax.set_ylabel("Ordered Residuals")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure("residuals_qq_simple.png")
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
        # Only fit in the plausible range (-10, 10) to avoid outliers skewing slope
        mask_fit = (theoretical_quantiles > -10) & (theoretical_quantiles < 10) & \
                   (sample_quantiles > -10) & (sample_quantiles < 10)
        
        if mask_fit.sum() > 10:
            slope_t, intercept_t = np.polyfit(theoretical_quantiles[mask_fit], sample_quantiles[mask_fit], 1)
            r_t = np.corrcoef(theoretical_quantiles[mask_fit], sample_quantiles[mask_fit])[0, 1]
        else:
            slope_t, intercept_t, r_t = 0, 0, 0
        
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
        # ref_line = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
        ref_line = np.array([-15, 15]) # Fixed ref line
        ax.plot(ref_line, ref_line, color='#e74c3c', linestyle='--', 
                linewidth=2, alpha=0.8, label='Perfect fit (y=x)', zorder=2)
        
        # Data points - layered transparency
        ax.scatter(theoretical_quantiles, sample_quantiles, c='#2ecc71', s=12, 
                   alpha=0.4, edgecolors='none', zorder=3)
        step = max(1, n_samples // 100)
        ax.scatter(theoretical_quantiles[::step], sample_quantiles[::step], 
                   c='#2ecc71', s=20, alpha=0.9, 
                   edgecolors='#1a7a3e', linewidths=0.5, zorder=4)
        
        ax.set_xlabel("Theoretical Quantiles", fontsize=11)
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
        
        # Fixed limits to match histogram
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        print("[Eval] Skipping QQ-plot (scipy not available).")
        
    # --- Advanced Residual Diagnostics (Conditional Checks) ---
    print("\n[Eval] Generating Advanced Residual Diagnostics (Conditional Checks)...")
    
    # Check if we captured variance and indices
    if 'var_log_eval' in locals() and var_log_eval is not None:
        sigma_log_eval = np.sqrt(var_log_eval)
        std_resid = resid_log / sigma_log_eval
        
        # 1. Calibration Table
        print("\n=== Calibration Metrics (Empirical Coverage) ===")
        print("Interval  | Nominal | Empirical | Gap")
        print("----------|---------|-----------|-----")
        for alpha in [0.50, 0.80, 0.95]:
            # Central interval z-score
            z_score = stats.norm.ppf(0.5 + alpha/2)
            lower = mu_log_eval - z_score * sigma_log_eval
            upper = mu_log_eval + z_score * sigma_log_eval
            covered = (y_true_log_eval >= lower) & (y_true_log_eval <= upper)
            empirical = covered.mean()
            print(f"{int(alpha*100)}% CI    | {alpha:.3f}   | {empirical:.3f}     | {empirical-alpha:+.3f}")
            
        # 2. Standardized Residuals (Homoskedasticity check)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(mu_log_eval, std_resid, alpha=0.1, s=2, color='gray')
        
        # Bin mean/std
        # Bin by predicted value
        bins = np.linspace(mu_log_eval.min(), mu_log_eval.max(), 20)
        bin_idx = np.digitize(mu_log_eval, bins)
        bin_means = [std_resid[bin_idx == i].mean() for i in range(1, len(bins))]
        bin_stds = [std_resid[bin_idx == i].std() for i in range(1, len(bins))]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', color='red', 
                    label='Binned Mean ± 1 Std', capsize=3)
        
        ax.axhline(0, color='black', linestyle='--')
        ax.axhline(1, color='green', linestyle=':', label='Ideal Std=1')
        ax.axhline(-1, color='green', linestyle=':')
        
        ax.set_xlabel("Predicted Log Price")
        ax.set_ylabel("Standardized Residual $z = (y - \mu) / \sigma$")
        ax.set_title("Standardized Residuals vs Prediction (Homoskedasticity Check)", fontweight='bold')
        ax.legend()
        save_figure("residuals_standardized_vs_pred.png")
        plt.show()
        
        # 3. PIT Histogram (Uniformity check)
        # Assuming Gaussian likelihood (default)
        pit_values = stats.norm.cdf(y_true_log_eval, loc=mu_log_eval, scale=sigma_log_eval)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pit_values, bins=20, density=True, color='purple', alpha=0.6, edgecolor='white')
        ax.axhline(1.0, color='black', linestyle='--', label='Ideal Uniform')
        ax.set_xlabel("PIT Value $u = F(y)$")
        ax.set_ylabel("Density")
        ax.set_title("PIT Histogram (Calibration Check)", fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend()
        save_figure("residuals_pit_histogram.png")
        plt.show()

    # 4. Conditional Bias Check (Residual vs Pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mu_log_eval, resid_log, alpha=0.05, s=2, color='#34495e')
    
    # Binned mean residual
    bins = np.linspace(mu_log_eval.min(), mu_log_eval.max(), 20)
    bin_idx = np.digitize(mu_log_eval, bins)
    bin_res_means = [resid_log[bin_idx == i].mean() for i in range(1, len(bins))]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    ax.plot(bin_centers, bin_res_means, 'r-o', linewidth=2, label='Binned Mean Residual')
    ax.axhline(0, color='black', linestyle='--')
    
    ax.set_xlabel("Predicted Log Price")
    ax.set_ylabel("Residual (Log Space)")
    ax.set_title("Conditional Bias: Residual vs Prediction", fontweight='bold')
    ax.legend()
    save_figure("residuals_vs_pred_bias.png")
    plt.show()

    # 5. Segment Stability Checks (if metadata available)
    if 'eval_indices' in locals() and eval_indices is not None:
        try:
            # Reconstruct evaluation dataframe subset
            # We need to handle duplicate indices if any, but loc might return duplicates
            # Safer to verify lengths
            print(f"[Eval] Segment Check: Indices length {len(eval_indices)}, Residuals length {len(resid_log)}")
            
            # Extract metadata
            # Warning: this relies on df_pred being available globally
            meta_subset = df_pred.loc[eval_indices].copy()
            
            # Ensure alignment (if loc returned different count due to non-unique indices)
            # This is a known risk. If huge mismatch, skip.
            if len(meta_subset) == len(resid_log):
                meta_subset['residual'] = resid_log
                
                # --- By Year Built ---
                year_col = next((c for c in ['year_built', 'yearbuilt', 'year'] if c in meta_subset.columns), None)
                if year_col:
                    meta_subset[year_col] = pd.to_numeric(meta_subset[year_col], errors='coerce')
                    valid_years = meta_subset.dropna(subset=[year_col])
                    valid_years = valid_years[(valid_years[year_col] > 1800) & (valid_years[year_col] <= 2025)]
                    
                    if len(valid_years) > 100:
                        # Bin by decade
                        valid_years['decade'] = (valid_years[year_col] // 10) * 10
                        decade_stats = valid_years.groupby('decade')['residual'].agg(['mean', 'count', 'std'])
                        decade_stats = decade_stats[decade_stats['count'] > 50] # Filter distinct decades
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.errorbar(decade_stats.index, decade_stats['mean'], 
                                    yerr=decade_stats['std'] / np.sqrt(decade_stats['count']),
                                    fmt='o-', color='teal', capsize=5)
                        ax.axhline(0, color='black', linestyle='--')
                        ax.set_xlabel("Decade Built")
                        ax.set_ylabel("Mean Residual ± SE")
                        ax.set_title("Residual Stability by Year Built (Time Check)", fontweight='bold')
                        save_figure("residuals_by_year.png")
                        plt.show()
                
                # --- By Building Class ---
                bldg_col = next((c for c in ["bldg_class", "building_class", "bldgclass"] if c in meta_subset.columns), None)
                if bldg_col:
                    # First letter only
                    meta_subset['class_major'] = meta_subset[bldg_col].astype(str).str[0].str.upper()
                    class_stats = meta_subset.groupby('class_major')['residual'].agg(['mean', 'count', 'std'])
                    class_stats = class_stats[class_stats['count'] > 50].sort_index()
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(class_stats.index, class_stats['mean'], yerr=class_stats['std'] / np.sqrt(class_stats['count']),
                           capsize=5, color='#e67e22', alpha=0.7)
                    ax.axhline(0, color='black', linewidth=1)
                    ax.set_xlabel("Building Class (Major)")
                    ax.set_ylabel("Mean Residual ± SE")
                    ax.set_title("Residual Stability by Building Class (Type Check)", fontweight='bold')
                    save_figure("residuals_by_bldg_class.png")
                    plt.show()
                    
            else:
                print("[Eval] Warning: Metadata index alignment failed. Skipping segment plots.")

        except Exception as e:
            print(f"[Eval] Segment analysis failed: {e}")
            
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
    
    # Compute per-dimension statistics
    z_stds = np.std(mu_z, axis=0)
    z_ranges = np.ptp(mu_z, axis=0)
    
    print(f"[Eval] Latent dimension stats (std):   {', '.join([f'z{i+1}={s:.3f}' for i, s in enumerate(z_stds)])}")
    print(f"[Eval] Latent dimension ranges (ptp): {', '.join([f'z{i+1}={r:.3f}' for i, r in enumerate(z_ranges)])}")
    
    # Debug model structure for Latent Importance Analysis
    if 'vae_model' in globals():
        print("\n[Eval] Inspecting VAE Model for Z->Y importance check:")
        # print(f"[Eval] Model type: {type(vae_model)}")
        print(f"[Eval] Model attributes: {[a for a in dir(vae_model) if not a.startswith('__')]}")
        # Check specific common names
        for attr in ['y_decoder', 'decoder_y', 'price_head', 'predictor', 'predict_y_from_z']:
            if hasattr(vae_model, attr):
                print(f"[Eval]   Found relevant attribute: {attr}")
    
    # Use FIXED axis limits to focus on core distribution, letting outliers fall outside
    # This focuses on the core structure rather than stretching to include outliers
    x_lim_fixed = (-0.1, 0.4)  # z1 axis
    y_lim_fixed = (-0.2, 0.7)  # z2 axis
    
    # Count how many points fall outside these limits
    outside_mask = (mu_z[:, 0] < -0.1) | (mu_z[:, 0] > 0.4) | (mu_z[:, 1] < -0.2) | (mu_z[:, 1] > 0.7)
    n_outside = outside_mask.sum()
    print(f"[Eval] Using fixed axis limits x={x_lim_fixed}, y={y_lim_fixed}. {n_outside} points ({100*n_outside/len(mu_z):.2f}%) fall outside.")
    
    # --- Latent Dimension Correlation Analysis ---
    # Analyze what each latent dimension correlates with to generate descriptive labels
    print("\n[Eval] Analyzing latent dimension correlations...")
    
    # Features to correlate with
    correlation_features = {}
    if log_y_col in df_pred.columns:
        correlation_features['Log Price'] = df_pred[log_y_col].astype(float).values
    if price_col in df_pred.columns:
        correlation_features['Price'] = df_pred[price_col].astype(float).values
    
    # Size features
    size_cols = ["gross_sqft", "gross_square_feet", "land_sqft", "land_square_feet", "sqft", "total_units"]
    size_col = next((c for c in size_cols if c in df_pred.columns), None)
    if size_col:
        correlation_features['Size'] = pd.to_numeric(df_pred[size_col], errors='coerce').values
    
    # Year features
    year_cols = ["year_built", "yearbuilt", "construction_year"]
    year_col = next((c for c in year_cols if c in df_pred.columns), None)
    if year_col:
        correlation_features['Year Built'] = pd.to_numeric(df_pred[year_col], errors='coerce').values
    
    # Compute correlations
    z1_correlations = {}
    z2_correlations = {}
    for feat_name, feat_values in correlation_features.items():
        # Only compute for valid (finite) values
        valid_mask = np.isfinite(feat_values) & np.isfinite(mu_z[:, 0])
        if valid_mask.sum() > 100:
            corr_z1 = np.corrcoef(mu_z[valid_mask, 0], feat_values[valid_mask])[0, 1]
            corr_z2 = np.corrcoef(mu_z[valid_mask, 1], feat_values[valid_mask])[0, 1]
            z1_correlations[feat_name] = corr_z1
            z2_correlations[feat_name] = corr_z2
            print(f"[Eval]   {feat_name}: z1 corr={corr_z1:.3f}, z2 corr={corr_z2:.3f}")
    
    # Auto-label based on top 3 correlated features
    def get_label_for_dim(corr_dict, dim_num):
        if not corr_dict:
            return f"z{dim_num}"
        # Sort features by absolute correlation (descending)
        sorted_feats = sorted(corr_dict.keys(), key=lambda k: abs(corr_dict[k]), reverse=True)
        top_3 = sorted_feats[:3]
        # Create label with top 3 features
        feat_strs = []
        for feat in top_3:
            corr = corr_dict[feat]
            if abs(corr) >= 0.1:  # Include if correlation is non-trivial
                direction = "+" if corr > 0 else "-"
                feat_strs.append(f"{direction}{feat}")
        if feat_strs:
            return f"z{dim_num} ({', '.join(feat_strs)})"
        else:
            return f"z{dim_num}"
    
    z1_label = get_label_for_dim(z1_correlations, 1)
    z2_label = get_label_for_dim(z2_correlations, 2)
    print(f"[Eval] Auto-generated latent labels: z1='{z1_label}', z2='{z2_label}'")
    
    # --- Encoder Weight Analysis ---
    # Trace back which INPUT features contribute most to each latent dimension
    print("\n[Eval] Analyzing encoder weights for input feature contributions...")
    try:
        # Get encoder weights from the model
        encoder_weights = None
        if hasattr(vae_model, 'encoder'):
            # Try to get the first linear layer weights
            for name, param in vae_model.encoder.named_parameters():
                if 'weight' in name and param.dim() == 2:
                    encoder_weights = param.detach().cpu().numpy()
                    print(f"[Eval]   Found encoder layer: {name}, shape={encoder_weights.shape}")
                    break
        
        if encoder_weights is not None and len(feature_names_x_out) > 0:
            # Get output weights from last encoder layer to latent mean
            latent_weights = None
            for name, param in vae_model.named_parameters():
                if 'mu' in name.lower() and 'weight' in name and param.dim() == 2:
                    latent_weights = param.detach().cpu().numpy()
                    print(f"[Eval]   Found latent mean layer: {name}, shape={latent_weights.shape}")
                    break
            
            if latent_weights is not None:
                # Analyze which input features have highest magnitude weights to each latent dim
                n_latents = min(2, latent_weights.shape[0])
                for lat_idx in range(n_latents):
                    weights_to_latent = np.abs(latent_weights[lat_idx, :])
                    # Sum up contribution through encoder if shapes allow
                    if encoder_weights.shape[0] == latent_weights.shape[1]:
                        # Direct connection: latent <- hidden <- input
                        input_contrib = np.abs(encoder_weights).sum(axis=0)
                    else:
                        input_contrib = weights_to_latent
                    
                    # Get top contributing features
                    if len(input_contrib) == len(feature_names_x_out):
                        top_k = min(5, len(feature_names_x_out))
                        top_indices = np.argsort(input_contrib)[-top_k:][::-1]
                        top_features = [(feature_names_x_out[i], input_contrib[i]) for i in top_indices]
                        print(f"[Eval]   z{lat_idx+1} top input features: {[(f, f'{w:.3f}') for f, w in top_features]}")
    except Exception as e:
        print(f"[Eval]   Encoder weight analysis failed: {e}")
    
    # --- SHAP Attribution (brief test) ---
    # Use SHAP KernelExplainer to attribute latent dimensions to input features
    print("\n[Eval] SHAP attribution for latent dimensions...")
    # --- Encoder Weight Analysis (SHAP) ---
    # "We computed SHAP attributions for the encoder latent means z_mu(x) to quantify which 
    # standardized input features most strongly influence each latent coordinate."
    # Important: Latent coordinates are not identifiable up to rotation. These are run-specific diagnostics.
    
    try:
        import shap
        # Use simple background from the STANDARDIZED input (X_all_np)
        # This matches exactly what the encoder receives.
        n_background = min(100, len(X_all_np))
        n_explain = min(100, len(X_all_np))
        
        # Sample background from standardized data
        rng_shap = np.random.default_rng(42)
        bg_idx = rng_shap.choice(len(X_all_np), n_background, replace=False)
        background = X_all_np[bg_idx]
        
        # Sample explain set
        explain_idx = rng_shap.choice(len(X_all_np), n_explain, replace=False)
        X_explain = X_all_np[explain_idx]
        
        # Define wrapper to get ONLY z1 and z2 (standardized input -> latent means)
        def encoder_z1z2(x_batch):
            # Check if x_batch is tensor or numpy
            if isinstance(x_batch, np.ndarray):
                x_batch = torch.from_numpy(x_batch).float().to(DEVICE)
            
            with torch.no_grad():
                # Encode to get mu, logvar
                mu, _ = vae_model.encode(x_batch)
                # Return only first 2 dimensions
                return mu[:, :2].cpu().numpy()

        import time
        start = time.time()
        
        # KernelSHAP on the z1,z2 projection
        explainer = shap.KernelExplainer(encoder_z1z2, background)
        shap_values = explainer.shap_values(X_explain)
        
        elapsed = time.time() - start
        print(f"[Eval]   SHAP completed in {elapsed:.1f}s for {n_explain} samples")
        
        # Handling SHAP output (list of arrays for multi-output)
        z1_shap, z2_shap = None, None
        
        if isinstance(shap_values, list) and len(shap_values) >= 2:
            z1_shap = shap_values[0]
            z2_shap = shap_values[1]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3: # (N, features, outputs)
                z1_shap = shap_values[:, :, 0]
                z2_shap = shap_values[:, :, 1]
            elif shap_values.ndim == 2:
                print("[Eval]   SHAP returned 2D array. Using as z1.")
                z1_shap = shap_values
        
        # Feature Renaming Map
        FEATURE_RENAMES = {
            'building_sales_sum': 'Bldg Sales Vol',
            'unit_sales_mean_roll2': 'Recent Trend',
            'unit_sales_mean_cum': 'Avg Cumul Sales',
            'unique_units': 'Unit Count',
            'log_gross_sqft': 'Log Size',
            'gross_sqft': 'Size',
            'year_built': 'Year Built',
            'log_sale_price': 'Log Price',
            'sale_price': 'Price'
        }
        
        def format_shap_label(z_shap, z_idx):
            if z_shap is None: return None
            
            z_imp = np.abs(z_shap).mean(axis=0)
            total_imp = z_imp.sum() + 1e-9
            
            top3_idx = np.argsort(z_imp)[-3:][::-1]
            
            parts = []
            for i in top3_idx:
                # Use feature_names_x_out which matches column order of X_all_np
                feat_raw = feature_names_x_out[i]
                feat_name = FEATURE_RENAMES.get(feat_raw, feat_raw) 
                pct = (z_imp[i] / total_imp) * 100
                parts.append(f"{pct:.0f}% {feat_name}")
            
            return f"z{z_idx}\n({', '.join(parts)})"

        if z1_shap is not None:
            shap_z1_label = format_shap_label(z1_shap, 1)
            if shap_z1_label: z1_label = shap_z1_label.replace("\n", " ")

        if z2_shap is not None:
            shap_z2_label = format_shap_label(z2_shap, 2)
            if shap_z2_label: z2_label = shap_z2_label.replace("\n", " ")
        
        if z1_shap is not None or z2_shap is not None:
            print(f"[Eval]   SHAP-based labels applied: '{z1_label}', '{z2_label}'")
            
            # Principled Redundancy Check: Latent Value Correlation
            z1_vals = mu_z[:, 0]
            z2_vals = mu_z[:, 1]
            latent_corr = np.corrcoef(z1_vals, z2_vals)[0, 1]
            
            print(f"[Eval]   Latent Coord Correlation (z1 vs z2): {latent_corr:.3f}")
            if abs(latent_corr) > 0.8:
                 print("[Eval]   WARNING: High correlation between latent coordinates. "
                       "Attributions may split across redundant axes.")
        else:
            print("[Eval]   Could not extract SHAP values.")

    except Exception as e:
        print(f"[Eval]   SHAP attribution failed: {e}")
        import traceback
        traceback.print_exc()

    # --- Latent Importance for Price Prediction (Permutation Test) ---
    print("\n[Eval] Calculating Latent Importance for Price Prediction (Z -> Y)...")
    try:
        # Check for price head
        if not hasattr(vae_model, 'price_mean_head'):
            print("[Eval]   No 'price_mean_head' found on model. Skipping Z->Y importance.")
        else:
            # 1. Baseline Performance (MSE/R2 using all Z)
            # Use full mu_z (random subset for speed if huge)
            n_perm = min(5000, len(mu_z))
            perm_idx = np.random.choice(len(mu_z), n_perm, replace=False)
            
            z_batch = mu_z[perm_idx]
            y_target = y_true_log_eval[perm_idx] if y_true_log_eval is not None else np.zeros(n_perm)
            
            # Helper to predict Y from Z
            def predict_y_from_z(z_in):
                t_z = torch.from_numpy(z_in).float().to(DEVICE)
                with torch.no_grad():
                    # Output of price_mean_head is usually (N, 1)
                    y_out = vae_model.price_mean_head(t_z)
                return y_out.cpu().numpy().ravel()
            
            y_pred_base = predict_y_from_z(z_batch)
            mse_base = np.mean((y_target - y_pred_base)**2)
            
            print(f"[Eval]   Baseline MSE (subset n={n_perm}): {mse_base:.4f}")
            
            # 2. Permute each dimension and measure MSE increase
            importances = []
            for dim_i in range(z_batch.shape[1]):
                z_permuted = z_batch.copy()
                # Shuffle ONLY this dimension
                np.random.shuffle(z_permuted[:, dim_i])
                
                y_pred_perm = predict_y_from_z(z_permuted)
                mse_perm = np.mean((y_target - y_pred_perm)**2)
                
                # Importance = Increase in MSE
                imp = mse_perm - mse_base
                importances.append(imp)
                print(f"[Eval]     z{dim_i+1} Importance (MSE increase): {imp:.4f}")
            
            # Normalize to percentages
            total_imp = sum(importances) + 1e-9
            print(f"[Eval]   Relative Importance: " + 
                  ", ".join([f"z{i+1}={100*imp/total_imp:.0f}%" for i, imp in enumerate(importances)]))
            
            if len(importances) >= 3 and importances[2] / total_imp > 0.10:
                print("[Eval]   NOTE: z3 has >10% importance. Consider visualizing it.")
                
    except Exception as e:
        print(f"[Eval]   Latent importance failed: {e}")


    
    
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
    
    # --- 7a-ORIG. ORIGINAL Simple latent scatter by price deciles (for comparison) ---
    log_price_all = df_pred[log_y_col].astype(float).values
    mask_finite_price = np.isfinite(log_price_all)
    
    if mask_finite_price.sum() > 0:
        # Compute decile edges on finite values
        decile_edges = np.quantile(log_price_all[mask_finite_price], np.linspace(0, 1, 11))
        decile_idx = np.full_like(log_price_all, fill_value=-1, dtype=int)
        decile_idx[mask_finite_price] = np.searchsorted(decile_edges[1:-1], 
                                                         log_price_all[mask_finite_price], side="right")
        valid_mask = decile_idx >= 0
        
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(mu_z[valid_mask, 0], mu_z[valid_mask, 1],
                        c=decile_idx[valid_mask], s=5, alpha=0.6, cmap="viridis")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Sale-price decile (0=lowest, 9=highest)")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title("Latent space (z1 vs z2) colored by sale-price deciles\n(Original Simple Version)",
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure("latent_space_price_decile_orig.png")
        plt.show()
    
    # --- 7a. Latent colored by SALE PRICE (continuous log scale) ---
    
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
        
        # Shuffle points to prevent ordering bias (last-plotted points appear on top)
        np.random.seed(42)  # Reproducible
        shuffle_idx = np.random.permutation(len(valid_log_price))
        z_shuffled = valid_z_price[shuffle_idx]
        price_shuffled = valid_log_price[shuffle_idx]
        
        # Scatter with continuous log-price coloring
        sc = ax.scatter(z_shuffled[:, 0], z_shuffled[:, 1], 
                        c=price_shuffled, s=8, alpha=0.5, cmap="viridis",
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
        
        ax.set_xlabel(z1_label, fontsize=11)
        ax.set_ylabel(z2_label, fontsize=11)
        
        # Apply fixed axis limits
        ax.set_xlim(x_lim_fixed)
        ax.set_ylim(y_lim_fixed)
        # ax.set_aspect('equal', adjustable='box')  # Commented: asymmetric limits
        
        # Title centered on axes
        ax.set_title("Latent Space by Price\nScatter with density contours", 
                     fontsize=12, fontweight='bold', pad=8)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        save_figure("latent_space_price.png")
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
        
        ax.set_xlabel(z1_label, fontsize=11)
        ax.set_ylabel(z2_label, fontsize=11)
        
        # Apply fixed axis limits
        ax.set_xlim(x_lim_fixed)
        ax.set_ylim(y_lim_fixed)
        # ax.set_aspect('equal', adjustable='box')  # Commented: asymmetric limits
        
        fig.suptitle("Latent Space by Price", fontsize=14, fontweight='bold', y=0.98)
        ax.set_title("Hexbin showing mean price per region", fontsize=9, color='gray', pad=3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure("latent_space_price_hexbin.png")
        plt.show()
    
    # --- 7a-bis. Hexbin density plot for latent space (count only) ---
    if mask_valid_price.sum() > 100:
        fig, ax = plt.subplots(figsize=(7, 6))
        
        hb = ax.hexbin(valid_z_price[:, 0], valid_z_price[:, 1], 
                       gridsize=40, cmap='Blues', mincnt=1, linewidths=0.2)
        
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label("Count", fontsize=10)
        
        ax.set_xlabel(z1_label, fontsize=11)
        ax.set_ylabel(z2_label, fontsize=11)
        
        # Apply fixed axis limits
        ax.set_xlim(x_lim_fixed)
        ax.set_ylim(y_lim_fixed)
        # ax.set_aspect('equal', adjustable='box')  # Commented: asymmetric limits
        
        fig.suptitle("Latent Space Density", fontsize=14, fontweight='bold', y=0.98)
        ax.set_title("Hexbin showing point concentration", fontsize=9, color='gray', pad=3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure("latent_space_density.png")
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
            
            ax.set_xlabel(z1_label, fontsize=11)
            ax.set_ylabel(z2_label, fontsize=11)
            
            # Apply fixed axis limits
            ax.set_xlim(x_lim_fixed)
            ax.set_ylim(y_lim_fixed)
            # ax.set_aspect('equal', adjustable='box')  # Commented: asymmetric limits
        
            fig.suptitle("Latent Space by Size", fontsize=14, fontweight='bold', y=0.98)
            ax.set_title(f"Scatter with density contours · {size_col}", fontsize=9, color='gray', pad=3)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure("latent_space_size.png")
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
            
            ax.set_xlabel(z1_label, fontsize=11)
            ax.set_ylabel(z2_label, fontsize=11)
            
            # Apply fixed axis limits
            ax.set_xlim(x_lim_fixed)
            ax.set_ylim(y_lim_fixed)
            # ax.set_aspect('equal', adjustable='box')  # Commented: asymmetric limits
        
            fig.suptitle("Latent Space by Size", fontsize=14, fontweight='bold', y=0.98)
            ax.set_title("Hexbin showing mean size per region", fontsize=9, color='gray', pad=3)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure("latent_space_size_hexbin.png")
            plt.show()
        else:
            print("[Eval] Not enough valid size data for plot.")
    else:
        print("[Eval] No size/sqft column found; skipping Size latent plot.")

    # --- 7c. Latent colored by Building Class (AGGREGATED) ---
    bldg_class_candidates = ["bldg_class", "building_class", "bldg_class_group", "bldgclass"]
    bldg_class_col = next((c for c in bldg_class_candidates if c in df_pred.columns), None)
    
    if bldg_class_col:
        print(f"[Eval] Using '{bldg_class_col}' for Building Class Viz.")
        
        # Filter to same price range as price plots (>= $100K)
        bldg_price_mask = mask_valid_price  # Reuse the mask from price plots
        print(f"[Eval] Building class plots using {bldg_price_mask.sum()} properties with price >= $100K")
        
        bldg_z = mu_z[bldg_price_mask]
        bldg_series_full = df_pred[bldg_class_col].astype(str)
        
        # --- 7c-ORIG. ORIGINAL Simple building class scatter (for comparison) ---
        bldg_series_raw = bldg_series_full[bldg_price_mask]
        codes_raw, uniques_raw = pd.factorize(bldg_series_raw)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        for code, label in enumerate(uniques_raw):
            mask = codes_raw == code
            if not np.any(mask):
                continue
            ax.scatter(bldg_z[mask, 0], bldg_z[mask, 1], s=5, alpha=0.5, label=label)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title("Latent space (z1 vs z2) colored by building class\n(Original Simple Version)",
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        n_classes = len(uniques_raw)
        if n_classes <= 10:
            ax.legend(title="Building class", fontsize=8)
        else:
            ax.legend(title="Building class (truncated)", fontsize=6, ncol=2)
        plt.tight_layout()
        save_figure("latent_space_bldg_orig.png")
        plt.show()
        
        # --- Enhanced version: Aggregate to first letter (major category) ---
        # Use the filtered data from above (same properties as price plots)
        bldg_series = bldg_series_full[bldg_price_mask].str[0].str.upper()
        
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
            ax.scatter(bldg_z[mask, 0], bldg_z[mask, 1], s=8, alpha=0.5, 
                      color=cmap(color_idx), label=lbl, edgecolors='none')
        
        ax.set_xlabel(z1_label, fontsize=11)
        ax.set_ylabel(z2_label, fontsize=11)
        
        # Apply fixed axis limits
        ax.set_xlim(x_lim_fixed)
        ax.set_ylim(y_lim_fixed)
        # ax.set_aspect('equal', adjustable='box')  # Commented: asymmetric limits
        
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
        save_figure("latent_space_bldg.png")
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
    save_figure("latent_marginals.png")
    plt.show()

print("[Eval] All visualizations completed.")
