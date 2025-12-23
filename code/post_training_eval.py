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
# CONTRACTS & HELPERS (Mandatory Gates)
# ----------------------------------------------------------
def first_present(cols, df):
    for c in cols:
        if c in df.columns:
            return c
    return None

def digitize_safe(x, edges):
    idx = np.digitize(x, edges, right=False)
    return np.clip(idx, 1, len(edges) - 1)

from torch.distributions import Normal
_STD_NORMAL = Normal(torch.tensor(0.0), torch.tensor(1.0))

def normal_cdf_np(z_in):
    zt = torch.as_tensor(np.asarray(z_in, dtype=np.float32))
    return _STD_NORMAL.cdf(zt).cpu().numpy()

def normal_ppf_scalar(p_in):
    pt = torch.tensor(float(p_in), dtype=torch.float32)
    return float(_STD_NORMAL.icdf(pt).cpu().item())

def apply_global_price_filter(y_true_log, y_true, mu_log, var_log, eval_indices, eval_pos_idx, min_price_log):
    """
    Apply global price filter UNIVERSALLY to all array-likes.
    """
    y_true_log = np.asarray(y_true_log).reshape(-1)
    mask = y_true_log >= float(min_price_log)

    n_before = int(mask.size)
    n_after = int(mask.sum())
    if n_after < n_before:
        print(f"[Eval] Global filter: log(price) >= {float(min_price_log):.6f}. Kept {n_after}/{n_before}.")
    else:
        print(f"[Eval] Global filter: kept all {n_before} samples.")

    def _f(a):
        if a is None:
            return None
        a = np.asarray(a)
        if len(a) != n_before:
             # Try to catch mismatch early, unless variable length list
             pass 
        return a[mask]

    return _f(y_true_log), _f(y_true), _f(mu_log), _f(var_log), _f(eval_indices), _f(eval_pos_idx)

@torch.no_grad()
def predict_mu_from_z(vae_model, zt):
    out = vae_model.price_mean_head(zt)
    if hasattr(vae_model, "price_mean_output") and callable(getattr(vae_model, "price_mean_output")):
        out = vae_model.price_mean_output(out)
    return out

@torch.no_grad()
def check_head_vs_trainer_mu(vae_model, vae_trainer, X_all_np, eval_pos_idx, n=2048, atol=1e-5):
    """
    Gate: Ensure vae_model.price_mean_head(z) matches trainer.predict(x)
    before trusting any Z-based attribution.
    """
    vae_model.eval()
    if eval_pos_idx is None or len(eval_pos_idx) == 0:
        raise RuntimeError("eval_pos_idx is empty; cannot gate head vs trainer mapping.")

    idx = eval_pos_idx[: min(n, len(eval_pos_idx))]
    xb = torch.from_numpy(X_all_np[idx].astype(np.float32, copy=False)).to(DEVICE)

    # 1. Encode -> Z -> Head -> y
    mu_z, _ = vae_model.encode(xb)
    mu_head = predict_mu_from_z(vae_model, mu_z).detach().cpu().numpy().reshape(-1)

    # 2. Trainer End-to-End
    mu_tr, _ = vae_trainer.predict_price_and_uncertainty(xb, batch_size=1024)
    mu_tr = np.asarray(mu_tr).reshape(-1)

    max_abs = float(np.max(np.abs(mu_head - mu_tr)))
    mean_abs = float(np.mean(np.abs(mu_head - mu_tr)))
    print(f"[Contract] Head vs Trainer Check: max_diff={max_abs:.6f}, mean_diff={mean_abs:.6f}")
    assert max_abs < atol, f"Mismatch (max={max_abs}): do not interpret Z importance or Z SHAP until this passes."
    
# ----------------------------------------------------------

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
    
    # Load and force EVAL mode
    vae_model.load_state_dict(state_dict)
    vae_model.to(DEVICE)
    vae_model.eval()
    print(f"[Eval] Loaded model_state_dict into VAE model from best_model.pth and set to EVAL mode.")
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
    print(f"Median |y - y_hat| (log):                              {median_abs_log:.4f}")
    print(f"95th pct |y - y_hat| (log):                            {p95_abs_log:.4f}")

# ----------------------------------------------------------
# 4. Metrics & Diagnostics Preparation
#    (Logic: CV if available, else Fallback. GLOBAL FILTER applied to both.)
# ----------------------------------------------------------

# A. CV Detection & Branching
pred_price_col_cv = f"cv_predicted_{price_col}"
pred_flag_col_cv  = "cv_prediction_available"

have_cv_point = (pred_price_col_cv in df_pred.columns) and (pred_flag_col_cv in df_pred.columns)

cv_mu_log_candidates = ["cv_mu_log", f"cv_mu_{log_y_col}", f"cv_predicted_{log_y_col}"]
cv_sigma_log_candidates = ["cv_sigma_log", f"cv_sigma_{log_y_col}", "cv_pred_uncertainty_log_price_std"]

cv_mu_log_col = first_present(cv_mu_log_candidates, df_pred)
cv_sigma_log_col = first_present(cv_sigma_log_candidates, df_pred)

# Fail-closed distribution check: need both mu and sigma from CV
have_cv_dist = have_cv_point and (cv_mu_log_col is not None) and (cv_sigma_log_col is not None)

if have_cv_point:
    print("\n[Eval] Found CV point predictions; using CV-stitched predictions for held-out metrics.")
    mask_pred = df_pred[pred_flag_col_cv].astype(bool).values
    df_eval = df_pred.loc[mask_pred].copy()
    print(f"[Eval] Rows with CV predictions: {df_eval.shape[0]} / {df_pred.shape[0]}")

    y_true_log_eval = df_eval[log_y_col].astype(float).values
    y_true_eval     = df_eval[price_col].astype(float).values
    y_pred_level    = df_eval[pred_price_col_cv].astype(float).values

    # Point estimate (mu)
    if cv_mu_log_col is not None:
        mu_log_eval = df_eval[cv_mu_log_col].astype(float).values
    else:
        # Fallback for point-only: log of the level prediction
        mu_log_eval = np.log(np.clip(y_pred_level, 1e-12, np.inf))

    var_log_eval = None
    if have_cv_dist:
        sigma_log = df_eval[cv_sigma_log_col].astype(float).values
        var_log_eval = np.square(np.clip(sigma_log, 1e-9, np.inf))
        
        # Calculate NLL/Metric with Prob
        _report_metrics(y_true_log_eval, y_true_eval, mu_log_eval, var_log_eval, label_prefix="CV stitched (mu_log + sigma_log)")
    else:
        print("\n=== Posterior Predictive Metrics (CV Point-Only) ===")
        print("[Eval] Missing paired (cv_mu_log, cv_sigma_log). Skipping NLL, coverage, PIT.")
        resid_log = y_true_log_eval - mu_log_eval
        rmse_log = float(np.sqrt(np.mean(resid_log ** 2)))
        mae_log  = float(np.mean(np.abs(resid_log)))
        rmse_price = float(np.sqrt(np.mean((y_true_eval - np.exp(mu_log_eval)) ** 2)))
        mae_price  = float(np.mean(np.abs(y_true_eval - np.exp(mu_log_eval))))
        print(f"RMSE (log-price): {rmse_log:.4f}")
        print(f"MAE  (log-price): {mae_log:.4f}")
        print(f"RMSE (price):     {rmse_price:,.4f}")
        print(f"MAE  (price):     {mae_price:,.4f}")

    eval_indices = df_eval.index.values
    # Indices in original df_pred
    eval_pos_idx = np.where(mask_pred)[0]

else:
    print("\n[Eval] cv_* columns not found; falling back to direct predictions from best_model on a random hold-out split.")
    if BEST_MODEL_PATH is None:
        raise RuntimeError("[Eval] No cv_* columns and no best_model.pth; cannot compute predictive metrics.")

    # mask for rows where the target is observed
    y_mask = y_mask_np.squeeze()
    obs_idx = np.where(y_mask > 0.0)[0]
    if obs_idx.size == 0:
        raise RuntimeError("[Eval] No observed targets according to VAE mask; cannot evaluate.")

    y_true_log_all = df_pred[log_y_col].astype(float).values[obs_idx]
    y_true_all     = df_pred[price_col].astype(float).values[obs_idx]
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
    
    # New semantics: var_log = exp(2 * log_sigma)
    var_log = np.exp(2 * log_var)

    print(f"[Eval] Evaluating on random hold-out: {len(test_rel_idx)} / {N_obs} observed rows.")
    _report_metrics(y_true_log, y_true, log_mu, var_log, label_prefix="best_model, random hold-out")
    
    # store for residual diagnostics
    y_true_log_eval = y_true_log
    y_true_eval = y_true
    mu_log_eval = log_mu
    var_log_eval = var_log

    # Indices for metadata fetch
    eval_indices = df_pred.index.values[obs_idx][test_rel_idx]
    eval_pos_idx = obs_idx[test_rel_idx]

# --- GLOBAL FILTER: Price >= $100k (Applied UNIVERSALLY) ---
MIN_PRICE_LOG_GLOBAL = np.log(100_000.0)

# Filter aligned arrays using the helper
y_true_log_eval, y_true_eval, mu_log_eval, var_log_eval, eval_indices, eval_pos_idx = apply_global_price_filter(
    y_true_log_eval, y_true_eval, mu_log_eval, var_log_eval, eval_indices, eval_pos_idx, MIN_PRICE_LOG_GLOBAL
)

# Also filter residuals to match
resid_log = y_true_log_eval - mu_log_eval

# optional metrics on "uncertainty OK" subset (re-check after filter)
if have_cv_dist and var_log_eval is not None:
    # We need aligned 'prediction_uncertainty_ok'
    # Difficult to align without mask matching. 
    # Simplified: skip or re-fetch based on eval_indices
    pass


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
    # WARNING: evidence suggests log_var is actually log_sigma (log std).
    # Over-coverage of ~20% with exp(log_var) supports this.
    # New semantics: var_log = exp(2 * log_sigma)
    var_log = np.exp(2 * log_var)

    print(f"[Eval] Evaluating on random hold-out: {len(test_rel_idx)} / {N_obs} observed rows.")
    _report_metrics(y_true_log, y_true, log_mu, var_log, label_prefix="best_model, random hold-out")



    # store for residual diagnostics
    y_true_log_eval = y_true_log
    y_true_eval = y_true # Fix NameError
    mu_log_eval = log_mu
    var_log_eval = var_log
    resid_log = y_true_log_eval - mu_log_eval # Fix IndexError/Stale Global
    # Indices for metadata fetch
    eval_indices = df_pred.index.values[obs_idx][test_rel_idx]
    # Integer positions in df_pred (for X_mask / mu_z alignment)
    # obs_idx are the positions in df_pred of observed rows
    # test_rel_idx are the positions in obs_idx
    eval_pos_idx = obs_idx[test_rel_idx]

    # --- GLOBAL FILTER: Price >= $100k ---
    # Apply UNIVERSALLY to all downstream diagnostics as requested.
    MIN_PRICE_LOG_GLOBAL = np.log(100_000)
    
    # Check if we have targets to filter
    if y_true_log_eval is not None:
        mask_global = y_true_log_eval >= MIN_PRICE_LOG_GLOBAL
        n_before = len(y_true_log_eval)
        n_after = mask_global.sum()
        
        if n_after < n_before:
            print(f"\n[Eval] Applying GLOBAL Price Filter (>= $100k). Kept {n_after}/{n_before} samples.")
            
            # Filter all core evaluation arrays
            y_true_log_eval = y_true_log_eval[mask_global]
            if y_true_eval is not None: y_true_eval = y_true_eval[mask_global]
            if mu_log_eval is not None: mu_log_eval = mu_log_eval[mask_global]
            if var_log_eval is not None: var_log_eval = var_log_eval[mask_global]
            if resid_log is not None: resid_log = resid_log[mask_global]
            
            # Filter indices (critical for metadata alignment)
            if eval_indices is not None: eval_indices = eval_indices[mask_global]
            if eval_pos_idx is not None: eval_pos_idx = eval_pos_idx[mask_global]
        else:
            print(f"\n[Eval] GLOBAL Price Filter (>= $100k) applied. All {n_before} samples valid.")

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
    # Footnote explaining loss function
    ax.text(0.5, -0.12, "Total Loss = Reconstruction Loss + beta * KL Divergence",
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
    ax.set_title("r = log(y_true) - log(y_hat)", fontsize=9, color='gray', pad=3)
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
                    alpha=0.7, label=f'nu={nu_ref:.1f}', zorder=1)
        
        # Best-fit Student-t: same gray color but SOLID line (not dashed)
        pdf_t_best = stats.t.pdf(x_grid, df=df_fit, loc=loc_fit, scale=scale_fit)
        ax.plot(x_grid, pdf_t_best, color='gray', linestyle='-', linewidth=2, 
                alpha=0.9, label=f'Student-t fit (nu={df_fit:.1f})', zorder=2)

    ax.set_xlabel("Residual", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    
    # Fixed axis limits for residual plots
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 2)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)  # subtle zero line
    
    # Title + subtitle with formula
    fig.suptitle("Residuals vs Heavy-Tailed References", fontsize=14, fontweight='bold', y=0.98)
    ax.set_title("r = log(y_true) - log(y_hat) * Log-transformed prices", 
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
                        label=f'95% CI (Student-t nu={df_fit:.1f})', zorder=1)
        
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
        ax.text(0.05, 0.95, f"Slope: {slope_t:.3f}\nR2: {r_t**2:.3f}",  
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
        
    # --- Helper: Safe Digitize ---
    def digitize_safe(x, edges):
        # np.digitize returns indices 1..len(edges)-1 normally
        # but 0 for < edges[0] and len(edges) for >= edges[-1]
        # We want to clamp to 1..len(edges)-1 to avoid index errors or dropouts
        idx = np.digitize(x, edges, right=False)
        return np.clip(idx, 1, len(edges)-1)

    # --- Helper: Log Price to Currency Ticks ---
    def get_log_price_ticks(min_log, max_log):
        # Extended range to cover likely values
        log_ticks = [np.log(100_000), np.log(250_000), np.log(500_000), 
                     np.log(1_000_000), np.log(2_500_000), np.log(5_000_000),
                     np.log(10_000_000), np.log(25_000_000), np.log(50_000_000),
                     np.log(100_000_000), np.log(250_000_000), np.log(500_000_000),
                     np.log(1_000_000_000)]
        log_labels = ['$100K', '$250K', '$500K', '$1M', '$2.5M', '$5M', '$10M', '$25M', '$50M',
                      '$100M', '$250M', '$500M', '$1B']
        
        valid_ticks = [(t, l) for t, l in zip(log_ticks, log_labels) if min_log <= t <= max_log]
        
        # Ensure we have at least min/max coverage if no standard ticks fall in range
        if not valid_ticks:
             valid_ticks = [(min_log, f"{np.exp(min_log)/1000:.0f}K"), (max_log, f"{np.exp(max_log)/1000:.0f}K")]
             
        return [t for t, l in valid_ticks], [l for t, l in valid_ticks]

    # --- Advanced Residual Diagnostics (Conditional Checks) ---
    print("\n[Eval] Generating Advanced Residual Diagnostics (Conditional Checks)...")
    
    # Check if we captured variance and indices
    # Check if we captured variance and indices
    if 'var_log_eval' in locals() and var_log_eval is not None:
        sigma_log_eval = np.sqrt(var_log_eval)
        std_resid = resid_log / sigma_log_eval
        
        # 1. Calibration Table (Scipy-free, using Torch for robust CDF/PPF)
        from torch.distributions import Normal
        _STD_NORMAL = Normal(torch.tensor(0.0), torch.tensor(1.0))

        def normal_cdf_np(z_in):
            # Safe numpy -> torch -> numpy cdf
            zt = torch.from_numpy(np.array(z_in, dtype=np.float32))
            return _STD_NORMAL.cdf(zt).cpu().numpy()

        def normal_ppf_scalar(p_in):
             pt = torch.tensor(float(p_in), dtype=torch.float32)
             return float(_STD_NORMAL.icdf(pt).cpu().item())

        print("\n=== Calibration Metrics (Empirical Coverage) ===")
        print("Interval  | Nominal | Empirical | Gap")
        print("----------|---------|-----------|-----")
        for alpha in [0.50, 0.80, 0.95]:
            # Central interval z-score
            z_score = normal_ppf_scalar(0.5 + alpha/2)
            lower = mu_log_eval - z_score * sigma_log_eval
            upper = mu_log_eval + z_score * sigma_log_eval
            covered = (y_true_log_eval >= lower) & (y_true_log_eval <= upper)
            empirical = covered.mean()
            print(f"{int(alpha*100)}% CI    | {alpha:.3f}   | {empirical:.3f}     | {empirical-alpha:+.3f}")
            
        # 2a. Standardized Residuals vs Prediction (Heteroskedasticity check)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(mu_log_eval, std_resid, alpha=0.1, s=2, color='gray')
        
        # Bin mean/std
        # Bin by predicted value
        bins = np.linspace(mu_log_eval.min(), mu_log_eval.max(), 20)
        bin_idx = digitize_safe(mu_log_eval, bins)
        
        bin_means = []
        bin_stds = []
        bin_centers = []
        
        for i in range(1, len(bins)):
            mask_bin = bin_idx == i
            if mask_bin.sum() > 5:
                bin_means.append(std_resid[mask_bin].mean())
                bin_stds.append(std_resid[mask_bin].std())
                bin_centers.append(0.5 * (bins[i-1] + bins[i]))
        
        if len(bin_centers) > 0:
            ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', color='red', 
                        label='Binned Mean +/- 1 Std', capsize=3)
        
        ax.axhline(0, color='black', linestyle='--')
        ax.axhline(1, color='green', linestyle=':', label='Ideal Std=1')
        ax.axhline(-1, color='green', linestyle=':')
        
        # Grid for bins
        for b in bins:
            ax.axvline(b, color='gray', linestyle=':', alpha=0.3)
            
        # Ticks: Log -> Currency (Vertical)
        curr_ticks, curr_labels = get_log_price_ticks(mu_log_eval.min(), mu_log_eval.max())
        ax.set_xticks(curr_ticks)
        ax.set_xticklabels(curr_labels, rotation=90)
        
        ax.set_xlabel("Predicted Price", fontsize=11)
        ax.set_ylabel("Residual (Standard Deviations)", fontsize=11)
        ax.set_title("Residuals vs Prediction (Normalized)", fontweight='bold')
        
        # Fixed Y-Range [-10, 10] as requested (or 20 for standardized?)
        # User said "make that range -10 10 for all residual plots"
        # Standardized might be larger, but let's stick to 10 for consistency if requested.
        ax.set_ylim(-10, 10)
        
        # Footnote
        plt.figtext(0.5, 0.01, "Values in Log Space. Y-axis in Standard Deviations.", 
                    ha="center", fontsize=9, fontstyle='italic')
        
        ax.legend()
        plt.tight_layout(rect=[0, 0.05, 1, 1]) # Space for footnote/vertical ticks
        save_figure("residuals_standardized_vs_pred.png")
        plt.show()

        # 2b. Standardized QQ Plot (Normality of conditional noise)
        if HAVE_SCIPY:
            fig, ax = plt.subplots(figsize=(6, 6))
            stats.probplot(std_resid, dist="norm", plot=ax)
            ax.set_title("QQ-Plot: Standardized Residuals vs Normal", fontsize=12, fontweight='bold')
            ax.set_ylabel("Ordered Standardized Residuals")
            # Add identity line
            ax.plot([-4, 4], [-4, 4], color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(-5, 5)
            ax.set_xlim(-5, 5)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_figure("residuals_standardized_qq.png")
            plt.show()

        # 2c. Absolute Residuals vs Prediction (Another Heteroskedasticity view)
        fig, ax = plt.subplots(figsize=(8, 5))
        abs_resid = np.abs(resid_log)
        ax.scatter(mu_log_eval, abs_resid, alpha=0.1, s=2, color='gray')
        
        # Smooth trend
        if len(mu_log_eval) > 100:
             bin_abs_means = []
             bin_centers_abs = []
             for i in range(1, len(bins)):
                 mask_bin = bin_idx == i
                 if mask_bin.sum() > 5:
                     bin_abs_means.append(abs_resid[mask_bin].mean())
                     bin_centers_abs.append(0.5 * (bins[i-1] + bins[i]))
             
             if len(bin_centers_abs) > 0:
                  ax.plot(bin_centers_abs, bin_abs_means, 'r-o', linewidth=2, label='Mean Abs Resid')
        
        ax.set_xlabel("Predicted Price")
        ax.set_ylabel("|Residual|")
        ax.set_title("Absolute Residuals vs Prediction", fontweight='bold')
        
        # Dollar Ticks Vertical
        ax.set_xticks(curr_ticks)
        ax.set_xticklabels(curr_labels, rotation=90)
        
        # Range [0, 10]
        ax.set_ylim(0, 10)
        
        plt.figtext(0.5, 0.01, "Values in Log Space", ha="center", fontsize=9, fontstyle='italic')
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_figure("residuals_absolute_vs_pred.png")
        plt.show()
        
        # 3. PIT Histogram (Probability Integral Transform) - Scipy Free
        # pit_values = F(y | x)
        # Using normal_cdf_np
        pit_values = normal_cdf_np((y_true_log_eval - mu_log_eval) / sigma_log_eval)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pit_values, bins=20, density=True, color='purple', alpha=0.6, edgecolor='white')
        ax.axhline(1.0, color='black', linestyle='--', label='Ideal Uniform')
        ax.set_xlabel("PIT Value $u = F(y)$")
        ax.set_ylabel("Density")
        ax.set_title("PIT Histogram", fontweight='bold')
        ax.text(0.5, 1.02, "Calibration Check", ha='center', va='bottom', transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_xlim(0, 1)
        ax.legend()
        save_figure("residuals_pit_histogram.png")
        plt.show()

    # 4. Conditional Bias Check (Residual vs Pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mu_log_eval, resid_log, alpha=0.05, s=2, color='#34495e')
    
    # Binned mean residual
    bins = np.linspace(mu_log_eval.min(), mu_log_eval.max(), 20)
    bin_idx = digitize_safe(mu_log_eval, bins)
    
    bin_res_means = []
    bin_centers = []
    
    for i in range(1, len(bins)):
        mask_bin = bin_idx == i
        if mask_bin.sum() > 5:
            bin_res_means.append(resid_log[mask_bin].mean())
            bin_centers.append(0.5 * (bins[i-1] + bins[i]))
    
    if len(bin_centers) > 0:
        ax.plot(bin_centers, bin_res_means, 'r-o', linewidth=2, label='Binned Mean Residual')
    ax.axhline(0, color='black', linestyle='--')
    
    # Grid
    for b in bins:
        ax.axvline(b, color='gray', linestyle=':', alpha=0.3)
        
    # Currency Ticks Vertical
    curr_ticks, curr_labels = get_log_price_ticks(mu_log_eval.min(), mu_log_eval.max())
    ax.set_xticks(curr_ticks)
    ax.set_xticklabels(curr_labels, rotation=90)
    
    ax.set_xlabel("Predicted Price")
    ax.set_ylabel("Residual")
    ax.set_title("Conditional Bias", fontweight='bold')
    
    # Range limits [-10, 10]
    ax.set_ylim(-10, 10)
    
    plt.figtext(0.5, 0.01, "Values in Log Space", ha="center", fontsize=9, fontstyle='italic')

    ax.legend()
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Margin for vertical ticks
    save_figure("residuals_vs_pred_bias.png")
    plt.show()



    # 5. Segment Stability Checks (if metadata available)
    if 'eval_indices' in locals() and eval_indices is not None:
        try:
            print(f"[Eval] Segment Check: Indices length {len(eval_indices)}, Residuals length {len(resid_log)}")
            
            # Use iloc with eval_pos_idx if available for safety, otherwise eval_indices
            if 'eval_pos_idx' in locals() and eval_pos_idx is not None:
                 meta_subset = df_pred.iloc[eval_pos_idx].copy()
            else:
                 meta_subset = df_pred.loc[eval_indices].copy()

            if len(meta_subset) == len(resid_log):
                meta_subset['residual'] = resid_log
                
                # --- By Sale Year (Time/Regime Drift) ---
                # Check for sale_year, sale_date, year_sale
                sale_col = next((c for c in ['sale_year', 'year_sale', 'saleyear', 'sale_date'] if c in meta_subset.columns), None)
                if sale_col:
                    # If date, extract year
                    if 'date' in sale_col.lower():
                        meta_subset[sale_col] = pd.to_datetime(meta_subset[sale_col], errors='coerce').dt.year
                    
                    meta_subset[sale_col] = pd.to_numeric(meta_subset[sale_col], errors='coerce')
                    
                    # Robust Sale Year Table Function
                    def sale_year_bin_table(meta_df, yr_col, y_true, mu, var, bin_width=1, min_count=20):
                        years = meta_df[yr_col].values
                        valid = np.isfinite(years)
                        years = years[valid].astype(int)
                        
                        y = y_true[valid]
                        mu_v = mu[valid]
                        resid = y - mu_v
                        
                        has_var = (var is not None)
                        if has_var:
                            var_v = np.clip(var[valid], 1e-9, np.inf)
                            sig_v = np.sqrt(var_v)
                        
                        start = (years.min() // bin_width) * bin_width
                        stop  = ((years.max() // bin_width) + 1) * bin_width
                        edges = np.arange(start, stop + bin_width, bin_width, dtype=int)
                        
                        idx = digitize_safe(years.astype(float), edges.astype(float))
                        
                        rows = []
                        for i in range(1, len(edges)):
                            m = (idx == i)
                            n = int(m.sum())
                            if n < min_count:
                                continue
                                
                            resid_i = resid[m]
                            mean_resid = float(np.mean(resid_i))
                            se_mean_resid = float(np.std(resid_i, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                            rmse_log = float(np.sqrt(np.mean(resid_i**2)))
                            
                            cov_50 = cov_80 = cov_95 = float('nan')
                            pit_mean = float('nan')
                            
                            if has_var:
                                y_i = y[m]
                                mu_i = mu_v[m]
                                sig_i = sig_v[m]
                                
                                # Coverage
                                for alpha, key in [(0.50, "cov_50"), (0.80, "cov_80"), (0.95, "cov_95")]:
                                    z = normal_ppf_scalar(0.5 + alpha/2.0)
                                    lower = mu_i - z * sig_i
                                    upper = mu_i + z * sig_i
                                    covered = float(np.mean((y_i >= lower) & (y_i <= upper)))
                                    if key == "cov_50": cov_50 = covered
                                    if key == "cov_80": cov_80 = covered
                                    if key == "cov_95": cov_95 = covered
                                
                                # PIT
                                pit = normal_cdf_np((y_i - mu_i) / sig_i)
                                pit_mean = float(np.mean(pit))
                            
                            rows.append({
                                "year": int(edges[i-1]),
                                "n": n,
                                "mean_resid": mean_resid,
                                "se_mean_resid": se_mean_resid,
                                "rmse_log": rmse_log,
                                "cov_50": cov_50,
                                "cov_80": cov_80,
                                "cov_95": cov_95,
                                "pit_mean": pit_mean
                            })
                        return pd.DataFrame(rows)

                    # Compute table
                    # Ensure alignment (y_true_log_eval is global filtered)
                    # meta_subset has already been filtered via iloc
                    
                    # We need strict alignment. meta_subset is from df_pred.iloc[eval_pos_idx] taking mask_global into account?
                    # Wait, eval_pos_idx was filtered by mask_global in step 1526.
                    # So meta_subset corresponds exactly to y_true_log_eval.
                    
                    tbl = sale_year_bin_table(meta_subset, sale_col, y_true_log_eval, mu_log_eval, var_log_eval)
                    
                    if not tbl.empty:
                        print("\n[Eval] Sale-year stratified metrics (1-year bins):")
                        print(tbl[['year', 'n', 'mean_resid', 'rmse_log', 'pit_mean', 'cov_50', 'cov_95']].head()) # Extended Print
                        
                        # Plot Mean Residual +/- SE
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.errorbar(tbl['year'], tbl['mean_resid'], yerr=tbl['se_mean_resid'], fmt='o-', color='purple', capsize=4, label='Mean Residual')
                        ax.axhline(0, color='black', linestyle='--')
                        
                        # Integers ticks
                        ax.set_xticks(tbl['year'])
                        ax.set_xticklabels(tbl['year'].astype(int), rotation=45)
                        
                        ax.set_xlabel("Sale Year")
                        ax.set_ylabel("Mean Residual +/- SE")
                        ax.set_title("Performance Stability by Sale Year", fontweight='bold')
                        plt.figtext(0.5, 0.01, "Values in Log Space", ha="center", fontsize=9, fontstyle='italic')
                        save_figure("residuals_by_sale_year.png")
                        plt.show()

                        # Plot Coverage (Un-Nested)
                        if 'cov_95' in tbl.columns and not tbl['cov_95'].isna().all():
                             fig, ax = plt.subplots(figsize=(10, 5))
                             ax.plot(tbl['year'], tbl['cov_50'], 'o-', label='50% CI')
                             ax.plot(tbl['year'], tbl['cov_80'], 'o-', label='80% CI')
                             ax.plot(tbl['year'], tbl['cov_95'], 'o-', label='95% CI')
                             
                             ax.axhline(0.50, color='gray', linestyle=':')
                             ax.axhline(0.80, color='gray', linestyle=':')
                             ax.axhline(0.95, color='gray', linestyle=':')
                             
                             ax.set_xticks(tbl['year'])
                             ax.set_xticklabels(tbl['year'].astype(int), rotation=45) # SALE YEAR IS ALREADY INT-LIKE
                             ax.set_xlabel("Sale Year")
                             ax.set_ylabel("Empirical Coverage")
                             ax.set_title("Uncertainty Calibration by Sale Year", fontweight='bold')
                             ax.legend()
                             save_figure("coverage_by_sale_year.png")
                             plt.show()

                # --- By Building Class (Residuals) ---
                bldg_class_candidates = ["bldg_class", "building_class", "bldg_class_group", "bldgclass"]
                bldg_col = next((c for c in bldg_class_candidates if c in meta_subset.columns), None)
                
                if bldg_col:
                    meta_subset[bldg_col] = meta_subset[bldg_col].astype(str).str.strip()
                    # Aggregate
                    grp = meta_subset.groupby(bldg_col)['residual']
                    
                    bldg_stats = pd.DataFrame({
                        'mean_resid': grp.mean(),
                        'std_resid': grp.std(),
                        'count': grp.count()
                    })
                    # Filter - Lower threshold to see more classes as requested
                    # Matched to latent plot or simply lower (e.g. 20)
                    bldg_stats = bldg_stats[bldg_stats['count'] > 20].sort_values('mean_resid')
                    
                    # Map codes to full names if possible
                    code_map = {
                        'A': '1-2 Family Houses', 'B': '2 Family Frame', 'C': 'Walk-up Apts', 
                        'D': 'Elevator Apts', 'R': 'Condominiums', 'S': 'Resid/Comm Mix',
                        'O': 'Office', 'K': 'Store/Loft', 'L': 'Loft', 'V': 'Vacant',
                        'P': 'Public', '01': '1 Fam', '02': '2 Fam', '03': '3 Fam'
                    }
                    
                    # Create full labels
                    bldg_stats['label'] = bldg_stats.index.to_series().apply(
                        lambda c: code_map.get(c[0].upper(), c) if len(c) > 0 else "Unknown"
                    )
                    
                    if not bldg_stats.empty:
                        print(f"\n[Eval] Residuals by Building Class (Top {len(bldg_stats)}):")
                        print(bldg_stats.head())
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        # Use integer x-axis for placing text
                        x_pos = np.arange(len(bldg_stats))
                        
                        ax.errorbar(x_pos, bldg_stats['mean_resid'], yerr=bldg_stats['std_resid'], 
                                    fmt='o', color='teal', capsize=5, label='Mean +/- 1 Std')
                        ax.axhline(0, color='black', linestyle='--')
                        
                        # Set limits to [-1, 1] usually enough for mean, but std error bars might exceed
                        # User wants no abbreviation to "resid"
                        ax.set_ylabel("Mean Residual +/- Std Dev")
                        ax.set_title("Performance by Building Class", fontweight='bold')
                        
                        # Replace X-axis ticks with Vertical Text Labels
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels([]) # Hide default labels
                        
                        for i, (idx, row) in enumerate(bldg_stats.iterrows()):
                            # Plot text vertically (bottom to top)
                            lbl = row['label']
                            # Place text slightly below axis or at the point? 
                            # Usually "tick labels" are below.
                            # User said "vertical axis tick labels rather than a b c etc"
                            # Standard solution: vertical rotation of x-labels
                            ax.text(i, ax.get_ylim()[0] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, 
                                    lbl, rotation=90, ha='center', va='top', fontsize=10, 
                                    transform=ax.transData)
                                    
                        # Or just use standard set_xticklabels with rotation=90?
                        # "vertical axis tick labels" -> yes, rotated ticks.
                        ax.set_xticklabels(bldg_stats['label'], rotation=90, fontsize=10)
                        
                        plt.figtext(0.5, 0.01, "Values in Log Space. Price >= $100k.", ha="center", fontsize=9, fontstyle='italic')
                        
                        # Ensure margins for tall labels
                        plt.figtext(0.5, 0.01, "Values in Log Space. Price >= $100k.", ha="center", fontsize=9, fontstyle='italic')
                        
                        # Ensure margins for tall labels
                        plt.tight_layout(rect=[0, 0.1, 1, 0.9]) # Extra bottom margin for vertical text
                        save_figure("residuals_by_bldg_class.png")
                        plt.show()
                        
                             # (Old nested coverage plot removed)

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
                        ax.set_title("Residual Stability by Year Built", fontweight='bold')
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
                    ax.set_title("Residual Stability by Building Class", fontweight='bold')
                    save_figure("residuals_by_bldg_class.png")
                    plt.show()
                    
            else:
                print("[Eval] Warning: Metadata index alignment failed. Skipping segment plots.")

        except Exception as e:
            print(f"[Eval] Segment analysis failed: {e}")

    if 'eval_pos_idx' in locals() and 'X_mask_np' in globals() and eval_pos_idx is not None:
        try:
            # Get mask for the evaluation subset
            # X_mask_np matches df_pred. Legend: 1=observed, 0=missing
            mask_subset = X_mask_np[eval_pos_idx]
            
            # Calculate fraction missing per row
            # If 1=observed, then mean() is fraction observed.
            # Fraction missing = 1.0 - mean()
            missing_frac = 1.0 - mask_subset.mean(axis=1)
            
            # Check for range
            max_miss = missing_frac.max()
            print(f"[Eval] Missingness check: Max missing frac = {max_miss:.3f}")
            
            if max_miss > 0.01:
                fig, ax = plt.subplots(figsize=(8, 5))
                # Binned analysis
                bins_m = np.linspace(0, max_miss + 0.01, 10)
                bin_idx_m = digitize_safe(missing_frac, bins_m)
                
                # Check for empty bins
                bin_centers_m = []
                bin_err_means = []
                bin_sigma_means = [] # If we had sigma
                
                for i in range(1, len(bins_m)):
                    mask_bin = bin_idx_m == i
                    if mask_bin.sum() > 5:
                        bin_centers_m.append(0.5 * (bins_m[i-1] + bins_m[i]))
                        bin_err_means.append(np.abs(resid_log)[mask_bin].mean())
                        
                if len(bin_centers_m) > 1:
                    fig, ax1 = plt.subplots(figsize=(8, 5))
                    ax1.plot(bin_centers_m, bin_err_means, 'r-o', label='Mean Abs Resid')
                    ax1.set_xlabel("Fraction of Features Missing")
                    ax1.set_ylabel("|Residual|", color='red')
                    ax1.tick_params(axis='y', labelcolor='red')
                    
                    # Add Sigma vs Missingness if available
                    if 'var_log_eval' in locals() and var_log_eval is not None:
                         # Recompute bins for sigma
                         bin_sigma_means = []
                         sigma_log = np.sqrt(np.clip(var_log_eval, 1e-9, np.inf))
                         for i in range(1, len(bins_m)):
                             mask_bin = bin_idx_m == i
                             if mask_bin.sum() > 5:
                                 bin_sigma_means.append(sigma_log[mask_bin].mean())
                         
                         if len(bin_sigma_means) == len(bin_centers_m):
                             ax2 = ax1.twinx()
                             ax2.plot(bin_centers_m, bin_sigma_means, 'b--s', label='Mean Sigma (Uncertainty)')
                             ax2.set_ylabel("Sigma", color='blue')
                             ax2.tick_params(axis='y', labelcolor='blue')
                             plt.figtext(0.5, 0.01, "Values in Log Space", ha="center", fontsize=9, fontstyle='italic')
                             
                             # Combined legend
                             lines1, labels1 = ax1.get_legend_handles_labels()
                             lines2, labels2 = ax2.get_legend_handles_labels()
                             ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                    else:
                        ax1.legend()
                    
                    ax1.set_title("Error & Uncertainty vs Missingness", fontweight='bold')
                    save_figure("error_vs_missingness.png")
                    plt.show()
                else:
                    print("[Eval] Not enough populated bins for natural missingness plot.")
                    
            else:
                print("[Eval] No natural missingness found (Max < 1%). Running SYNTHETIC Missingness Stress Test.")
                # Synthetic Test: artificially mask observed values and check degradation
                # We need access to X_obs (or similar).
                # We can use X_filled_np[eval_pos_idx] as "ground truth" X (assuming imputation is good or it's observed)
                # Actually, strictly we should use X_filled_np and force-mask it.
                
                try:
                    if 'vae_trainer' in locals() and 'X_filled_np' in globals():
                        X_eval_base = X_filled_np[eval_pos_idx]
                        
                        fracs_to_test = [0.0, 0.1, 0.2, 0.3, 0.5]
                        rmse_list = []
                        unc_list = []
                        
                        # Subsample for speed
                        n_syn = min(500, len(X_eval_base))
                        rng_syn = np.random.default_rng(99)
                        sub_idx = rng_syn.choice(len(X_eval_base), n_syn, replace=False)
                        X_sub = X_eval_base[sub_idx]
                        y_sub = y_true_log_eval[sub_idx]
                        
                        print(f"[Eval] Running Synthetic Stress Test on n={n_syn} samples...")
                        
                        for f in fracs_to_test:
                            # Create mask: 0=missing
                            # Keep (1-f) observed
                            mask_syn = rng_syn.binomial(1, 1-f, size=X_sub.shape).astype(np.float32)
                            
                            # Zero out missing values in input (assuming model expects 0 for missing)
                            X_sub_masked = X_sub * mask_syn
                            
                            # Convert to tensor
                            X_batch = torch.from_numpy(X_sub_masked).float().to(DEVICE)
                            # IMPORTANT: Ideally we pass the mask too. 
                            # If predict_price_and_uncertainty doesn't take mask, this is imperfect 
                            # but tests robustness to zero-imputation at least.
                            
                            # Log-capture silence
                            try:
                                log_mu_syn, log_var_syn = vae_trainer.predict_price_and_uncertainty(X_batch, batch_size=500)
                                log_mu_syn = np.asarray(log_mu_syn).reshape(-1)
                                
                                # Variance Semantics Fix: exp(2*log_var)
                                sigma_syn = np.exp(np.asarray(log_var_syn).reshape(-1))
                                
                                # Compute RMSE
                                rmse_syn = np.sqrt(np.mean((y_sub - log_mu_syn)**2))
                                mean_sigma = sigma_syn.mean()
                                
                                rmse_list.append(rmse_syn)
                                unc_list.append(mean_sigma)
                            except Exception as ex:
                                print(f"  Failed for f={f}: {ex}")
                                rmse_list.append(np.nan)
                                unc_list.append(np.nan)
                        
                        # Plot
                        fig, ax1 = plt.subplots(figsize=(8, 5))
                        ax1.plot(fracs_to_test, rmse_list, 'r-o', label='RMSE (Log)')
                        ax1.set_xlabel("Synthetic Missing Fraction")
                        ax1.set_ylabel("RMSE", color='red')
                        
                        ax2 = ax1.twinx()
                        ax2.plot(fracs_to_test, unc_list, 'b--s', label='Mean Predicted Sigma')
                        ax2.set_ylabel("Predicted Std Dev", color='blue')
                        
                        plt.title("Synthetic Missingness Stress Test", fontweight='bold')
                        save_figure("synthetic_missingness_stress.png")
                        plt.show()
                except Exception as e_syn:
                    print(f"[Eval] Synthetic test failed: {e_syn}")

                
        except Exception as e:
            print(f"[Eval] Missingness diagnostic failed: {e}")

    # 7. Spatial Diagnostics (Map of Residuals)
    # Check for coordinates in df_pred
    coord_cols = None
    if 'x_coord' in df_pred.columns and 'y_coord' in df_pred.columns:
        coord_cols = ('x_coord', 'y_coord')
    elif 'longitude' in df_pred.columns and 'latitude' in df_pred.columns:
        coord_cols = ('longitude', 'latitude')
    
    if 'eval_indices' in locals() and eval_indices is not None and coord_cols:
        try:
             # Use safe ILOC indexing with eval_pos_idx if available
             # This guarantees we pick exactly the rows corresponding to resid_log
             if 'eval_pos_idx' in locals() and eval_pos_idx is not None:
                 meta_subset = df_pred.iloc[eval_pos_idx].copy()
             else:
                 meta_subset = df_pred.loc[eval_indices].copy()
                 
             x_col, y_col = coord_cols
             
             # Filter to finite coordinates
             valid_geo = meta_subset.dropna(subset=[x_col, y_col])
             # Align residuals
             if len(valid_geo) > 100:
                  # We need to valid_geo vs resid_log alignment.
                  # Since valid_geo is a subset of meta_subset, we can join or index
                  # Easiest: add residual to meta_subset first
                  meta_subset['residual'] = resid_log
                  valid_geo = meta_subset.dropna(subset=[x_col, y_col])
                  
                  fig, ax = plt.subplots(figsize=(8, 8))
                  # Plot binned residuals (hexbin) to show spatial pattern
                  hb = ax.hexbin(valid_geo[x_col], valid_geo[y_col], C=valid_geo['residual'],
                                 gridsize=50, cmap='coolwarm', vmin=-1, vmax=1, reduce_C_function=np.mean)
                  plt.colorbar(hb, ax=ax, label="Mean Residual")
                  ax.set_title("Spatial Residual Map", fontweight='bold')
                  plt.figtext(0.5, 0.01, "Values in Log Space", ha="center", fontsize=9, fontstyle='italic')
                  
                  # Remove Lat/Long Ticks
                  ax.set_xticks([])
                  ax.set_yticks([])
                  ax.set_xlabel("")
                  ax.set_ylabel("")
                  
                  # Try to enforce aspect ratio if we assume lat/lon
                  ax.set_aspect('equal', adjustable='box')

                  # Add Basemap if possible
                  try:
                      import contextily as cx
                      # Assuming WGS84 (lat/lon) for data; contextily expects WebMercator usually,
                      # but we can try letting it reproject or specifying crs if supported.
                      # Safest basic usage for standard lat/lon data plotted on ax:
                      cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron)
                  except (ImportError, Exception) as e_map:
                      print(f"[Eval] Could not add basemap (contextily): {e_map}")
                  
                  save_figure("residuals_spatial_map.png")
                  plt.show()
             else:
                  print("[Eval] Not enough valid coordinates for spatial map.")
        except Exception as e:
            print(f"[Eval] Spatial diagnostic failed: {e}")

            
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
    
@torch.no_grad()
def perm_importance_groups(z, y, predict_fn, groups, n_repeats=10, seed=123):
    rng = np.random.default_rng(seed)
    y_pred_base = predict_fn(z)
    mse_base = float(np.mean((y - y_pred_base) ** 2))

    out = []
    for g in groups:
        deltas = []
        for _ in range(n_repeats):
            z_perm = z.copy()
            perm = rng.permutation(z.shape[0])
            z_perm[:, g] = z_perm[perm][:, g] # Permute group dimensions together
            y_pred = predict_fn(z_perm)
            mse = float(np.mean((y - y_pred) ** 2))
            deltas.append(mse - mse_base)
        
        out.append((g, float(np.mean(deltas)), float(np.std(deltas, ddof=1)) if n_repeats > 1 else 0.0))
    return mse_base, out

def top_corr_pair(z):
    c = np.corrcoef(z, rowvar=False)
    np.fill_diagonal(c, 0.0)
    i, j = np.unravel_index(np.argmax(np.abs(c)), c.shape)
    if i > j:
        i, j = j, i
    return [int(i), int(j)], float(c[i, j])

# ----------------------------------------------------
# HEAD vs TRAINER Contract Check
# ----------------------------------------------------
try:
    check_head_vs_trainer_mu(vae_model, vae_trainer, X_all_np, eval_pos_idx)
    
    # ----------------------------------------------------
    # Robust Z Importance (Group-Based)
    # ----------------------------------------------------
    # z_batch and y_target already aligned from latent extraction (X_filled_np -> mu_z)
    # We need Y aligned to mu_z? 
    # Current mu_z is all (len(X_filled_np))
    # y_target is needed. X_filled_np covers all predictions.
    # We should use observed subset for importance to validate against y_true_log_eval?
    # Actually, we can just use the indices we have.
    
    # Let's use the evaluation split for importance to be safe
    # eval_pos_idx -> indices into mu_z (X_filled_np)
    # y_true_log_eval -> targets
    
    if len(eval_pos_idx) > 5000:
        idx_imp = eval_pos_idx[:5000]
        y_imp = y_true_log_eval[:5000]
    else:
        idx_imp = eval_pos_idx
        y_imp = y_true_log_eval
        
    z_imp = mu_z[idx_imp]
    
    def predict_wrapper(z_in):
        zt = torch.from_numpy(z_in).float().to(DEVICE)
        return predict_mu_from_z(vae_model, zt).detach().cpu().numpy().reshape(-1)

    # Detect correlated pairs
    pair, pair_corr = top_corr_pair(z_imp)
    others = [k for k in range(z_imp.shape[1]) if k not in pair]
    groups = [[k] for k in range(z_imp.shape[1])] # Singles
    groups.append(pair) # Correlated Pair
    if len(others) > 0:
        groups.append(others) # Rest

    print(f"\n[Eval] Z Importance: Max |corr| pair for grouping: dims={pair}, corr={pair_corr:.6f}")

    mse_base, stats_g = perm_importance_groups(z_imp, y_imp, predict_wrapper, groups, n_repeats=5, seed=123)
    print(f"[Eval] Baseline MSE (on subset): {mse_base:.4f}")
    
    print("\n[Eval] Latent Importance (Delta MSE):")
    for g, mean_d, std_d in stats_g:
        print(f"  Group {g}: delta_MSE={mean_d:.4f} +/- {std_d:.4f}")

except AssertionError as e_gate:
    print(f"\n[Eval] SKIP Z Importance: {e_gate}")
except Exception as e_imp:
    print(f"\n[Eval] Z Importance Analysis failed: {e_imp}")
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
    


    # --- Latent Importance for Price Prediction (Permutation Test) ---
    print("\n[Eval] Calculating Latent Importance for Price Prediction (Z -> Y)...")
    
    # Default plot dimensions (if importance fails)
    plot_dim1 = 0
    plot_dim2 = 1
    
    try:
        # Check for price head
        if not hasattr(vae_model, 'price_mean_head'):
            print("[Eval]   No 'price_mean_head' found on model. Skipping Z->Y importance.")
        else:
            # 1. Baseline Performance (MSE/R2 using all Z)
            # Use eval subset only to match y_true_log_eval size
            # We now have eval_pos_idx which are integer positions in mu_z (if mu_z covers full dataset)
            
            if 'eval_pos_idx' in locals() and eval_pos_idx is not None:
                # Use only the evaluation set for this test
                z_eval_subset = mu_z[eval_pos_idx]
                y_target_subset = y_true_log_eval # Already aligned with eval_pos_idx
                
                # Subsample if too large
                n_perm = min(5000, len(z_eval_subset))
                perm_sub_idx = np.random.choice(len(z_eval_subset), n_perm, replace=False)
                
                z_batch = z_eval_subset[perm_sub_idx]
                y_target = y_target_subset[perm_sub_idx]
                y_target = y_target_subset[perm_sub_idx]
            else:
                 print("[Eval] Warning: eval_pos_idx not found. SKIPPING Z->Y Test to avoid index mismatch.")
                 # Removed unsafe fallback
                 z_batch = None
            
            if z_batch is not None:
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
                
                # Update plotting dimensions to top 2 important latents
                top_indices = np.argsort(importances)[::-1] # Descending importance
                plot_dim1 = top_indices[0]
                plot_dim2 = top_indices[1]
                print(f"[Eval]   Updated plotting axes to Top-2 contributors: z{plot_dim1+1} (x) and z{plot_dim2+1} (y)")
                
    except Exception as e:
        print(f"[Eval]   Latent importance failed: {e}")

    label_x = f"z{plot_dim1+1}"
    label_y = f"z{plot_dim2+1}"
    
    # Calculate robust default axis limits (global)
    # This ensures they are defined even if subsets are empty
    x_lim_fixed = np.percentile(mu_z[:, plot_dim1], [0.5, 99.5])
    y_lim_fixed = np.percentile(mu_z[:, plot_dim2], [0.5, 99.5])
    
    # --- SHAP Attribution (Running on Selected Dimensions) ---
    print(f"\n[Eval] SHAP attribution for selected latent dimensions: {label_x}, {label_y}...")
    try:
        import shap
        # Use simple background from the STANDARDIZED input (X_all_np)
        n_background = min(100, len(X_all_np))
        n_explain = min(100, len(X_all_np))
        
        rng_shap = np.random.default_rng(42)
        bg_idx = rng_shap.choice(len(X_all_np), n_background, replace=False)
        background = X_all_np[bg_idx]
        
        explain_idx = rng_shap.choice(len(X_all_np), n_explain, replace=False)
        X_explain = X_all_np[explain_idx]
        
        # Define wrapper to get ONLY the selected dimensions
        def encoder_subset(x_batch):
            if isinstance(x_batch, np.ndarray):
                x_batch = torch.from_numpy(x_batch).float().to(DEVICE)
            with torch.no_grad():
                mu, _ = vae_model.encode(x_batch)
                # Select the dimensions we strictly care about
                return mu[:, [plot_dim1, plot_dim2]].cpu().numpy()

        import time
        start = time.time()
        
        explainer = shap.KernelExplainer(encoder_subset, background)
        shap_values = explainer.shap_values(X_explain)
        
        elapsed = time.time() - start
        print(f"[Eval]   SHAP completed in {elapsed:.1f}s")
        
        shap_x, shap_y = None, None
        
        if isinstance(shap_values, list) and len(shap_values) >= 2:
            shap_x = shap_values[0]
            shap_y = shap_values[1]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3: 
                shap_x = shap_values[:, :, 0]
                shap_y = shap_values[:, :, 1]
            elif shap_values.ndim == 2:
                shap_x = shap_values
        
        # Feature Renaming Map (same as before)
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
        
        def format_shap_label(z_shap, dim_idx):
            if z_shap is None: return None
            z_imp = np.abs(z_shap).mean(axis=0)
            total_imp = z_imp.sum() + 1e-9
            top3_idx = np.argsort(z_imp)[-3:][::-1]
            parts = []
            for i in top3_idx:
                feat_raw = feature_names_x_out[i]
                feat_name = FEATURE_RENAMES.get(feat_raw, feat_raw) 
                pct = (z_imp[i] / total_imp) * 100
                parts.append(f"{pct:.0f}% {feat_name}")
            return f"z{dim_idx+1}\n({', '.join(parts)})"

        if shap_x is not None:
            lbl = format_shap_label(shap_x, plot_dim1)
            if lbl: label_x = lbl.replace("\n", " ")

        if shap_y is not None:
            lbl = format_shap_label(shap_y, plot_dim2)
            if lbl: label_y = lbl.replace("\n", " ")
        
        print(f"[Eval]   SHAP-based labels: '{label_x}', '{label_y}'")

    except Exception as e:
        print(f"[Eval]   SHAP attribution failed: {e}")


    
    
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
        sc = ax.scatter(mu_z[valid_mask, plot_dim1], mu_z[valid_mask, plot_dim2],
                        c=decile_idx[valid_mask], s=5, alpha=0.6, cmap="viridis")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Sale-price decile (0=lowest, 9=highest)")
        ax.set_xlabel(f"z{plot_dim1+1}")
        ax.set_ylabel(f"z{plot_dim2+1}")
        ax.set_title(f"Latent space (z{plot_dim1+1} vs z{plot_dim2+1}) colored by sale-price deciles\n(Original Simple Version)",
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure("latent_space_price_decile_orig.png")
        plt.show()
    
    # --- 7a. Latent colored by SALE PRICE (continuous log scale) ---
    
    # Filter: require price >= $100,000 (log >= 11.51) to exclude anomalies
    MIN_PRICE_LOG = np.log(100_000)  # ~11.51
    mask_valid_price = np.isfinite(log_price_all) & (log_price_all >= MIN_PRICE_LOG)
    
    # Define Axis Limits based on selected dimensions (Robust 99% interval)
    # This replaces any hardcoded limits
    if mask_valid_price.sum() > 100:
        z_vals_all = mu_z[mask_valid_price]
        x_lim_fixed = np.percentile(z_vals_all[:, plot_dim1], [0.5, 99.5])
        y_lim_fixed = np.percentile(z_vals_all[:, plot_dim2], [0.5, 99.5])
        # Add slight buffer
        x_pad = (x_lim_fixed[1] - x_lim_fixed[0]) * 0.1
        y_pad = (y_lim_fixed[1] - y_lim_fixed[0]) * 0.1
        x_lim_fixed = (x_lim_fixed[0]-x_pad, x_lim_fixed[1]+x_pad)
        y_lim_fixed = (y_lim_fixed[0]-y_pad, y_lim_fixed[1]+y_pad)
        
        print(f"[Eval] Dynamic axis limits for z{plot_dim1+1}/z{plot_dim2+1}: X={x_lim_fixed}, Y={y_lim_fixed}")
    
    print(f"[Eval] Price filter: {mask_valid_price.sum()} / {np.isfinite(log_price_all).sum()} " +
          f"properties with sale price >= $100K")
    
    if mask_valid_price.sum() > 100:
        valid_log_price = log_price_all[mask_valid_price]
        valid_z_price = mu_z[mask_valid_price]
        
        # --- VERSION 1: Scatter + Contour ---
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Add density contours first (behind scatter)
        add_density_contours(valid_z_price[:, plot_dim1], valid_z_price[:, plot_dim2], ax, 
                            levels=6, color='white', alpha=0.7)
        
        # Shuffle points to prevent ordering bias (last-plotted points appear on top)
        np.random.seed(42)  # Reproducible
        shuffle_idx = np.random.permutation(len(valid_log_price))
        z_shuffled = valid_z_price[shuffle_idx]
        price_shuffled = valid_log_price[shuffle_idx]
        
        # Scatter with continuous log-price coloring
        sc = ax.scatter(z_shuffled[:, plot_dim1], z_shuffled[:, plot_dim2], 
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
        
        ax.set_xlabel(label_x, fontsize=11)
        ax.set_ylabel(label_y, fontsize=11)
        
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
        
        hb = ax.hexbin(valid_z_price[:, plot_dim1], valid_z_price[:, plot_dim2], 
                       C=valid_log_price, reduce_C_function=np.mean,
                       gridsize=40, cmap='viridis', mincnt=1, linewidths=0.2)
        
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label("Mean Sale Price", fontsize=10)
        
        # Same log-scale labels
        if valid_ticks:
            cbar.set_ticks([t for t, l in valid_ticks])
            cbar.set_ticklabels([l for t, l in valid_ticks])
        
        ax.set_xlabel(label_x, fontsize=11)
        ax.set_ylabel(label_y, fontsize=11)
        
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
        
        hb = ax.hexbin(valid_z_price[:, plot_dim1], valid_z_price[:, plot_dim2], 
                       gridsize=40, cmap='Blues', mincnt=1, linewidths=0.2)
        
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label("Count", fontsize=10)
        
        ax.set_xlabel(label_x, fontsize=11)
        ax.set_ylabel(label_y, fontsize=11)
        
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
            
            add_density_contours(valid_z_size[:, plot_dim1], valid_z_size[:, plot_dim2], ax,
                                levels=6, color='white', alpha=0.7)
            
            sc = ax.scatter(valid_z_size[:, plot_dim1], valid_z_size[:, plot_dim2], c=log_size, s=8, alpha=0.5, 
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
            
            ax.set_xlabel(label_x, fontsize=11)
            ax.set_ylabel(label_y, fontsize=11)
            
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
            
            hb = ax.hexbin(valid_z_size[:, plot_dim1], valid_z_size[:, plot_dim2], 
                           C=log_size, reduce_C_function=np.mean,
                           gridsize=40, cmap='magma', mincnt=1, linewidths=0.2)
            
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label("Mean Sq Ft", fontsize=10)
            if valid_sqft:
                cbar.set_ticks([t for t, l in valid_sqft])
                cbar.set_ticklabels([l for t, l in valid_sqft])
            
            ax.set_xlabel(label_x, fontsize=11)
            ax.set_ylabel(label_y, fontsize=11)
            
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
            ax.scatter(bldg_z[mask, plot_dim1], bldg_z[mask, plot_dim2], s=5, alpha=0.5, label=label)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_title(f"Latent space (z{plot_dim1+1} vs z{plot_dim2+1}) colored by building class\n(Original Simple Version)",
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
        
        # Plot in predefined order for color consistency, legend outside
        for color_idx, letter in enumerate(present_classes):
            mask = (bldg_series == letter).values
            if not np.any(mask): 
                continue
            
            lbl_full = class_labels.get(letter, letter)
            ax.scatter(bldg_z[mask, plot_dim1], bldg_z[mask, plot_dim2], s=8, alpha=0.5, 
                      color=cmap(color_idx), label=lbl_full, edgecolors='none')
        
        ax.set_xlabel(label_x, fontsize=11)
        ax.set_ylabel(label_y, fontsize=11)
        
        # Apply fixed axis limits
        ax.set_xlim(x_lim_fixed)
        ax.set_ylim(y_lim_fixed)
        
        ax.set_title("Latent Space by Building Class", fontsize=12, fontweight='bold', pad=8)
        
        # Legend outside
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                  framealpha=0.9, title="Building Class")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Make room for legend
        save_figure("latent_space_bldg.png")
        plt.show()

    plt.show()
    
    # --- 7d-NEW. Latent Marginal Distributions (Uniform Scale) ---
    # Plot top 3 dimensions with uniform scaling for comparability
    n_dims = min(mu_z.shape[1], 3)
    fig, axes = plt.subplots(1, n_dims, figsize=(4*n_dims, 3.5), sharey=True, sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    # Compute global limit for x-axis
    z_global_min = np.percentile(mu_z[:, :n_dims], 0.5)
    z_global_max = np.percentile(mu_z[:, :n_dims], 99.5)
    pad = (z_global_max - z_global_min) * 0.1
    xlim_global = (z_global_min - pad, z_global_max + pad)

    for d, ax in enumerate(axes):
        ax.hist(mu_z[:, d], bins=40, alpha=0.8, color='#3498db', 
                edgecolor='white', linewidth=0.3, density=True)
        ax.set_xlabel(f"Latent Dim {d+1}", fontsize=10)
        if d == 0:
            ax.set_ylabel("Density", fontsize=10)
        
        ax.set_xlim(xlim_global)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle("Latent Marginal Distributions (Uniform Scale)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure("latent_marginals.png")
    plt.show()

print("[Eval] All visualizations completed.")
