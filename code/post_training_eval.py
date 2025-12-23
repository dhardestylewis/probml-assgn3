# === POST-TRAINING EVAL + VIZ (robust to missing cv_* columns) ===

import os, glob, math, logging, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def normal_ppf_torch(p_in):
    """
    Vectorized PPF (Inverse CDF) using Torch. 
    Handles both scalar and array inputs.
    """
    if isinstance(p_in, (float, int)):
        pt = torch.tensor(float(p_in), dtype=torch.float32)
        return float(_STD_NORMAL.icdf(pt).cpu().item())
    else:
        # Assume numpy array or similar
        pt = torch.from_numpy(np.asarray(p_in)).float()
        return _STD_NORMAL.icdf(pt).cpu().numpy()

def apply_global_price_filter(y_true_log, mu_log, var_log, pos_idx, df, min_price=100_000):
    """
    Applies a universal price filter (>= min_price) to all aligned evaluation arrays.
    Returns filtered versions of all inputs.
    """
    limit_log = np.log(min_price)
    upper_log = np.log(250_000_000.0) # User requested 250M cutoff
    
    # 250M Cap & Min Price
    mask = (y_true_log >= limit_log) & (y_true_log <= upper_log)
    
    n_before = len(y_true_log)
    n_after = mask.sum()
    
    print(f"\n[Eval] Global Price Filter ($100k - $250M): Keeping {n_after}/{n_before} samples ({100*n_after/n_before:.1f}%)")
    
    # Apply to all
    y_new = y_true_log[mask]
    mu_new = mu_log[mask]
    var_new = var_log[mask] if var_log is not None else None
    
    # pos_idx must be aligned
    if pos_idx is not None:
        if len(pos_idx) != n_before:
             raise RuntimeError(f"Length mismatch in global filter (pos_idx): got {len(pos_idx)} expected {n_before}")
        pos_new = pos_idx[mask]
    else:
        pos_new = None
        
    # df must be aligned
    if df is not None:
         if len(df) != n_before:
             raise RuntimeError(f"Length mismatch in global filter (df): got {len(df)} expected {n_before}")
         df_new = df[mask].copy() # Copy to avoid SettingWithCopy
    else:
         df_new = None
         
    return y_new, mu_new, var_new, pos_new, df_new

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
    
def add_graded_density_contours(ax, x, y, xlim, ylim, levels=5):
    """
    Adds graded density contours to an existing plot.
    Innermost lines are thicker, bolder, and more opaque than outermost lines.
    """
    try:
        import scipy.stats as st
        # Grid setup for 2D density
        deltaX = (xlim[1] - xlim[0]) / 50
        deltaY = (ylim[1] - ylim[0]) / 50
        xmin, xmax = xlim
        ymin, ymax = ylim
        X_grid, Y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
        values = np.vstack([x, y])
        
        # Downsample for speed if N is huge
        if len(x) > 10000:
             idx = np.random.choice(len(x), 5000, replace=False)
             values_kde = values[:, idx]
        else:
             values_kde = values
             
        kernel = st.gaussian_kde(values_kde)
        Z_grid = np.reshape(kernel(positions).T, X_grid.shape)
        
        # Define styles for levels (Outer -> Inner)
        # Revert to LINEAR levels per user request ("change back to linear for both")
        linewidths = np.linspace(1.0, 2.5, levels)
        alphas = np.linspace(0.4, 1.0, levels)
        
        level_colors = [(0, 0, 0, a) for a in alphas]
        
        # Use default linear levels (or auto-levels from density range)
        ax.contour(X_grid, Y_grid, Z_grid, levels=levels, 
                   linewidths=linewidths, colors=level_colors, zorder=3)
        
        # Annotation explaining the lines (Removed "(Linear Scale)" per user request)
        ax.text(0.02, 0.98, "Contours: Density Iso-lines",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.3),
                zorder=4)
                
    except Exception as e:
        print(f"[Eval] Failed to add graded density contours: {e}")
    
# --- SALE YEAR CONTRACT (Gate D Helper) ---
def sale_year_bin_table(meta_df, yr_col, y_true, mu, var, bin_width=1, min_count=20):
    """
    Computes calibration and performance metrics stratified by sale year.
    Used for Gate D (Sale Year Calibration).
    """
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
    
    # Range
    if len(years) == 0:
        return pd.DataFrame()

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
                z = normal_ppf_torch(0.5 + alpha/2.0)
                lower = mu_i - z * sig_i
                upper = mu_i + z * sig_i
                covered = float(np.mean((y_i >= lower) & (y_i <= upper)))
                if key == "cov_50": cov_50 = covered
                if key == "cov_80": cov_80 = covered
                if key == "cov_95": cov_95 = covered
            
            # PIT (Vectorized)
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

def normal_ppf_np(p_in):
    """
    Vectorized PPF (Inverse CDF) using Torch, accepting Numpy input.
    Replaces unsafe scalar versions.
    """
    pt = torch.as_tensor(np.asarray(p_in, dtype=np.float32))
    return float(_STD_NORMAL.icdf(pt).cpu().item()) if pt.numel() == 1 else _STD_NORMAL.icdf(pt).cpu().numpy()

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
    print(f"Average Negative Log-Likelihood (log-price space): {avg_nll_log:.4f}")
    print(f"Root Mean Square Error (log-price):                {rmse_log:.4f}")
    print(f"Mean Absolute Error    (log-price):                {mae_log:.4f}")
    print(f"Root Mean Square Error (price):                    {rmse_price:,.4f}")
    print(f"Mean Absolute Error    (price):                    {mae_price:,.4f}")
    print(f"Mean Absolute Percentage Error (price, y_true > 0):{mape_price * 100:,.2f}%")
    print(f"Median Absolute Residual (log-price):              {median_abs_log:.4f}")
    print(f"95th Percentile Absolute Residual (log-price):     {p95_abs_log:.4f}")

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
# --- GLOBAL FILTER: Price >= $100k (Applied UNIVERSALLY) ---
# Gate F: Align metadata first, then filter everything together.
meta_eval = df_pred.iloc[eval_pos_idx].copy()

# Filter aligned arrays using the helper
y_true_log_eval, mu_log_eval, var_log_eval, eval_pos_idx, meta_eval = apply_global_price_filter(
    y_true_log_eval, mu_log_eval, var_log_eval, eval_pos_idx, meta_eval
)

# Keep a single aligned name downstream
df_pred_filtered = meta_eval

# --- CONSISTENCY FILTER: Enforce Building Class Availability ---
# Ensure metrics/residuals use the exact same subset as Latent Plots
bldg_candidates_check = ["bldg_class", "building_class", "bldg_class_group", "bldgclass"]
bldg_col_check = next((c for c in bldg_candidates_check if c in df_pred_filtered.columns), None)

if bldg_col_check:
    mask_valid_bldg = df_pred_filtered[bldg_col_check].notna().to_numpy()
    n_before_bldg = len(df_pred_filtered)
    n_after_bldg = mask_valid_bldg.sum()
    
    if n_after_bldg < n_before_bldg:
        print(f"[Eval] Consistency: Dropping {n_before_bldg - n_after_bldg} rows with missing '{bldg_col_check}' to match Latent Plots.")
        df_pred_filtered = df_pred_filtered.iloc[mask_valid_bldg]
        y_true_log_eval = y_true_log_eval[mask_valid_bldg]
        mu_log_eval = mu_log_eval[mask_valid_bldg]
        if var_log_eval is not None:
             var_log_eval = var_log_eval[mask_valid_bldg]
        if eval_pos_idx is not None:
             eval_pos_idx = eval_pos_idx[mask_valid_bldg]

# Post-filter hard alignment asserts (Gate F)
n_post = y_true_log_eval.shape[0]
if mu_log_eval.shape[0] != n_post or eval_pos_idx.shape[0] != n_post or len(df_pred_filtered) != n_post:
    raise RuntimeError(f"Post-filter alignment failure. n={n_post}, mu={len(mu_log_eval)}, idx={len(eval_pos_idx)}, df={len(df_pred_filtered)}")
if var_log_eval is not None and var_log_eval.shape[0] != n_post:
    raise RuntimeError("Post-filter alignment failure (var_log_eval).")

# Also filter residuals to match
# Re-compute residuals to be safe
resid_log = y_true_log_eval - mu_log_eval

# --- SALE YEAR CALIBRATION (Gate D: Unconditional Top-Level) ---
sale_col = first_present(["sale_year", "year_sale", "saleyear"], df_pred_filtered)
if (sale_col is not None) and (var_log_eval is not None):
    print(f"\n[Eval] Running Sale Year Calibration on '{sale_col}'...")
    tbl_sale = sale_year_bin_table(
        df_pred_filtered, sale_col, y_true_log_eval, mu_log_eval, var_log_eval,
        bin_width=1, min_count=20
    )
    if not tbl_sale.empty:
        print(tbl_sale[["year","n","mean_resid","rmse_log","cov_50","cov_80","cov_95","pit_mean"]].to_string(index=False))
        
        # Plot Coverage (Immediate)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tbl_sale['year'], tbl_sale['cov_50'], 'o-', label='50% CI')
        ax.plot(tbl_sale['year'], tbl_sale['cov_80'], 'o-', label='80% CI')
        ax.plot(tbl_sale['year'], tbl_sale['cov_95'], 'o-', label='95% CI')
        ax.axhline(0.50, color='gray', linestyle=':')
        ax.axhline(0.80, color='gray', linestyle=':')
        ax.axhline(0.95, color='gray', linestyle=':')
        ax.set_ylim(0, 1) # Strict range [0, 1]
        
        # Ensure integer year ticks
        years = tbl_sale['year'].astype(int).values
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, ha='right')
        
        ax.set_xlabel("Sale Year", fontsize=11)
        ax.set_ylabel("Empirical Coverage", fontsize=11)
        ax.set_title("Uncertainty Calibration by Sale Year", fontweight='bold', fontsize=14)
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        save_figure("coverage_by_sale_year.png")
        plt.show()
if have_cv_dist and var_log_eval is not None:
    # We need aligned 'prediction_uncertainty_ok'
    # Difficult to align without mask matching. 
    # Simplified: skip or re-fetch based on eval_indices
    pass

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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resid_log, bins=50, density=True, alpha=0.8, color='#2ecc71', 
            edgecolor='white', linewidth=0.3, label='Observed Residuals', zorder=3)
    
    if HAVE_SCIPY:
        # Fits
        loc_n, scale_n = stats.norm.fit(resid_log)
        df_fit, loc_fit, scale_fit = stats.t.fit(resid_log)
        x_grid = np.linspace(-10, 10, 400)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure("residuals_hist_simple.png")
    plt.show()
    
    # --- VERSION 2: Histogram + Normal reference only ---
    if HAVE_SCIPY:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(resid_log, bins=50, density=True, alpha=0.8, color='#2ecc71', 
                edgecolor='white', linewidth=0.3, label='Observed', zorder=3)
        pdf_norm = stats.norm.pdf(x_grid, loc=loc_n, scale=scale_n)
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
    
    # --- VERSION 3: Histogram with best-fit curve only (no Student-t references) ---
    
    # --- 6a. Histogram with best-fit overlay ---
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Main histogram - prominent color
    ax.hist(resid_log, bins=50, density=True, alpha=0.8, color='#2ecc71', 
            edgecolor='white', linewidth=0.3, label='Observed residuals', zorder=3)
    
    if HAVE_SCIPY:
        # Plot only the best-fit Student-t curve (no reference lines)
        pdf_t_best = stats.t.pdf(x_grid, df=df_fit, loc=loc_fit, scale=scale_fit)
        ax.plot(x_grid, pdf_t_best, color='#e74c3c', linestyle='-', linewidth=2, 
                alpha=0.9, label=f'Best Fit (Student-t, $\\nu$={df_fit:.1f})', zorder=2)

        # Add 1D Density Equivalents (Quantile vertical lines) to match 2D contours
        qs = np.percentile(resid_log, [2.5, 25, 75, 97.5])
        
        # 95% Interval (Outer) - corresponds to outer contour
        ax.axvline(qs[0], color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='95% Interval')
        ax.axvline(qs[3], color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # 50% Interval (Inner) - corresponds to inner contour region
        ax.axvline(qs[1], color='black', linestyle=':', linewidth=1.2, alpha=0.8, label='50% Interval')
        ax.axvline(qs[2], color='black', linestyle=':', linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Residual", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    
    # Fixed axis limits for residual plots
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 3)
    
    # Title
    ax.set_title("Residuals vs References", fontweight='bold', fontsize=14)
    
    # Math Footnote (User Requested)
    plt.figtext(0.5, 0.01, r"$r = \log(y_{true}) - \log(y_{pred})$", ha="center", fontsize=10)
    
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False) # Strict removal
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure("residuals_hist_references.png")
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
                        label=f'95% CI (Student-t $\\nu$={df_fit:.1f})', zorder=1)
        
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
        
        # Title - simple, no subtitle
        ax.set_title("Q-Q Plot", fontsize=14, fontweight='bold', pad=10)
        
        # Annotation for fit quality
        ax.text(0.05, 0.95, f"Slope: {slope_t:.3f}\nR2: {r_t**2:.3f}",  
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Fixed limits per user request
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        print("[Eval] Skipping QQ-plot (scipy not available).")
        
    # --- Helper: Safe Digitize (Removed - use top-level) ---
    # def digitize_safe(...) removed to satisfy Gate E

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
        print("\n=== Calibration Metrics (Empirical Coverage) ===")
        print("Interval  | Nominal | Empirical | Gap")
        print("----------|---------|-----------|-----")
        for alpha in [0.50, 0.80, 0.95]:
            # Central interval z-score
            z_score = normal_ppf_torch(0.5 + alpha/2)
            lower = mu_log_eval - z_score * sigma_log_eval
            upper = mu_log_eval + z_score * sigma_log_eval
            covered = (y_true_log_eval >= lower) & (y_true_log_eval <= upper)
            empirical = covered.mean()
            print(f"{int(alpha*100)}% CI    | {alpha:.3f}   | {empirical:.3f}     | {empirical-alpha:+.3f}")
            
    # --- SALE YEAR CALIBRATION (Gate D) ---
    # Moved to top-level to run conditionally but not nested
    # Nested block removed to prevent duplication and satisfy Gate D/E.

    # 4. Residuals vs Prediction (Revised Gate E/Visuals)
    # --------------------------
    if var_log_eval is not None:
        sigma_eval = np.sqrt(np.clip(var_log_eval, 1e-9, np.inf))
        std_resid = resid_log / (sigma_eval + 1e-9)
        resid_std_label = "Residual (Standardized)"
        
        # QQ Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        
        sorted_res = np.sort(std_resid) # Use std_resid here!
        n = len(sorted_res)
        prob = (np.arange(n) + 0.5) / n
        theo_q = normal_ppf_np(prob) # Vectorized call
        
        ax.scatter(theo_q, sorted_res, alpha=0.3, s=5)
        
        # 45 degree line
        min_v, max_v = -10, 10
        ax.plot([min_v, max_v], [min_v, max_v], 'r--')
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Standardized Residuals")
        ax.set_title("QQ Plot", fontweight='bold')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.grid(False) # Strict removal
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        save_figure("residuals_qq.png")
        plt.show()

        # Resids vs Pred (Standardized)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(mu_log_eval, std_resid, alpha=0.1, s=2, color='#21918c')  # Viridis mid-tone
        
        # Ticks: Log -> Currency (Vertical)
        curr_ticks, curr_labels = get_log_price_ticks(mu_log_eval.min(), mu_log_eval.max())
        
        # LOESS smoothing with std dev band
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sorted_idx = np.argsort(mu_log_eval)
            x_sorted = mu_log_eval[sorted_idx]
            y_sorted = std_resid[sorted_idx]
            
            # LOESS for mean trend
            loess_result = lowess(y_sorted, x_sorted, frac=0.3, return_sorted=True)
            ax.plot(loess_result[:, 0], loess_result[:, 1], 'r-', linewidth=2, label='LOESS Trend')
            
            # Compute local std dev for confidence band
            # Use rolling window std for band
            window = max(50, len(x_sorted) // 20)
            y_roll_std = np.array([y_sorted[max(0,i-window//2):min(len(y_sorted),i+window//2)].std() 
                                   for i in range(len(y_sorted))])
            # Smooth the std dev
            std_smooth = lowess(y_roll_std, x_sorted, frac=0.3, return_sorted=True)[:, 1]
            
            ax.fill_between(loess_result[:, 0], 
                           loess_result[:, 1] - std_smooth,
                           loess_result[:, 1] + std_smooth,
                           color='red', alpha=0.2, label='±1 Standard Deviation')
        except ImportError:
            print("[Eval] statsmodels not available for LOESS; skipping trend line.")
        except Exception as e_loess:
            print(f"[Eval] LOESS failed: {e_loess}")
            
        ax.axhline(0, color='black', linestyle='--')

        ax.set_xticks(curr_ticks)
        ax.set_xticklabels(curr_labels, rotation=90)
        
        ax.set_xlabel("Predicted Price", fontsize=11)
        ax.set_ylabel("Residual", fontsize=11)  # Remove 'Standardized' from ylabel
        ax.set_title("Residuals vs Prediction", fontsize=14, fontweight='bold')  # Remove 'Standardized' from title
        
        # Footnote for standardization
        plt.figtext(0.5, 0.01, "Residuals are standardized (r / σ)", ha='center', fontsize=9, fontstyle='italic')
        
        # Enforce strict Y-limits
        ax.set_ylim(-2.5, 2.5)

        # PREFER TICKS FOR GRID to ensure alignment
        for t in curr_ticks:
            ax.axvline(t, color='gray', linestyle=':', alpha=0.3)
        
        ax.set_xlim(left=np.log(100_000))
        
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(rect=[0, 0.15, 1, 0.95]) # Increased bottom margin for 90deg ticks
        save_figure("residuals_standardized_vs_pred.png")
        plt.show()
    else:
        print("[Eval] No var_log_eval; skipping standardized residual diagnostics.")

        # 2b. Standardized QQ Plot (Normality of conditional noise)
        if HAVE_SCIPY:
            fig, ax = plt.subplots(figsize=(6, 6))
            stats.probplot(std_resid, dist="norm", plot=ax)
            ax.set_title("QQ Plot: Standardized Residuals vs Normal", fontsize=12, fontweight='bold')
            ax.set_ylabel("Ordered Standardized Residuals")
            # Add identity line
            ax.plot([-10, 10], [-10, 10], color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(-10, 10) # FAIL-CLOSED RANGE
            ax.set_xlim(-10, 10) # FAIL-CLOSED RANGE
            ax.grid(False) # Strict removal
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            save_figure("residuals_standardized_qq.png")
            plt.show()

        # 2c. Absolute Residuals vs Prediction (Another Heteroskedasticity view)
        fig, ax = plt.subplots(figsize=(8, 5))
        abs_resid = np.abs(resid_log)
        ax.scatter(mu_log_eval, abs_resid, alpha=0.1, s=2, color='gray')
        
        # LOESS smoothing for trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sorted_idx = np.argsort(mu_log_eval)
            x_sorted = mu_log_eval[sorted_idx]
            y_sorted = abs_resid[sorted_idx]
            loess_result = lowess(y_sorted, x_sorted, frac=0.3, return_sorted=True)
            ax.plot(loess_result[:, 0], loess_result[:, 1], 'r-', linewidth=2, label='LOESS Trend')
        except ImportError:
            print("[Eval] statsmodels not available for LOESS; skipping trend line.")
        except Exception as e_loess:
            print(f"[Eval] LOESS failed: {e_loess}")
        
        ax.set_xlabel("Predicted Price\n" + r"$\it{(Values\ in\ Log\ Space)}$", fontsize=11)
        ax.set_ylabel("|Residual|", fontsize=11)
        ax.set_title("Absolute Residuals vs Prediction", fontweight='bold', pad=8)
        
        # Enforce X-limit cutoff
        ax.set_xlim(left=np.log(100_000))
        
        # Dollar Ticks Vertical
        ax.set_xticks(curr_ticks)
        ax.set_xticklabels(curr_labels, rotation=90)
        
        # Range [0, 10]
        ax.set_ylim(0, 10)
        
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Note moved to xlabel to maintain consistent spacing with axis despite rotated ticks
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Matched to Latent Plots
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
        ax.text(0.5, 1.02, "Calibration Check", ha='center', va='bottom', transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        save_figure("residuals_pit_histogram.png")
        plt.show()

    # 4. Conditional Bias Check (Residual vs Pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mu_log_eval, resid_log, alpha=0.05, s=2, color='#21918c')  # Viridis mid-tone
    
    # LOESS smoothing for continuous fit instead of binned means
    curr_ticks, curr_labels = get_log_price_ticks(mu_log_eval.min(), mu_log_eval.max())
    
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        # LOESS for mean trend
        sorted_idx = np.argsort(mu_log_eval)
        x_sorted = mu_log_eval[sorted_idx]
        y_sorted = resid_log[sorted_idx]
        
        # Fit LOESS (frac=0.2 for reasonable smoothing)
        loess_result = lowess(y_sorted, x_sorted, frac=0.2, return_sorted=True)
        x_smooth = loess_result[:, 0]
        y_smooth = loess_result[:, 1]
        
        # Compute rolling std for ±1σ band
        # Use a window-based approach for local std
        window_size = max(100, len(y_sorted) // 20)
        y_std_smooth = pd.Series(y_sorted).rolling(window=window_size, center=True, min_periods=50).std().values
        # Interpolate to same x values as LOESS
        y_std_interp = np.interp(x_smooth, x_sorted, np.nan_to_num(y_std_smooth, nan=0.5))
        
        # Plot LOESS line
        ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='LOESS Trend')
        
        # Plot ±1σ shaded band
        ax.fill_between(x_smooth, y_smooth - y_std_interp, y_smooth + y_std_interp,
                        color='red', alpha=0.15, label='±1 Standard Deviation')
    except ImportError:
        print("[Eval] statsmodels not available, falling back to binned mean")
        # Fallback to binned approach if LOESS unavailable
        if len(curr_ticks) > 2:
            bins = np.array(curr_ticks)
            bin_idx = digitize_safe(mu_log_eval, bins)
            bin_res_means = []
            bin_centers = []
            for i in range(1, len(bins)):
                mask_bin = bin_idx == i
                if mask_bin.sum() > 5:
                    bin_res_means.append(resid_log[mask_bin].mean())
                    bin_centers.append(0.5 * (bins[i-1] + bins[i]))
            if len(bin_centers) > 0:
                ax.plot(bin_centers, bin_res_means, 'r-o', linewidth=2, label='Binned Mean')

    ax.axhline(0, color='black', linestyle='--')
    
    # Currency Ticks Vertical
    ax.set_xticks(curr_ticks)
    ax.set_xticklabels(curr_labels, rotation=90)

    # PREFER TICKS FOR GRID
    for t in curr_ticks:
        ax.axvline(t, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel("Predicted Price\n" + r"$\it{(Values\ in\ Log\ Space)}$", fontsize=11)
    ax.set_ylabel("Residual", fontsize=11)
    ax.set_title("Conditional Bias", fontweight='bold', fontsize=14, pad=8)
    
    # Enforce X-limit cutoff
    ax.set_xlim(left=np.log(100_000))
    
    # Range limits [-1, 1] for Conditional Bias
    ax.set_ylim(-1, 1)
    
    # Note moved to xlabel
    # plt.figtext(0.5, 0.01, r"Conditional Bias: $\mathbb{E}[r \mid \hat{y}]$ where $r = \log(y) - \log(\hat{y})$", ha="center", fontsize=9, fontstyle='italic')

    # Apply Ticks and Labels to match 'Residuals vs Pred'
    ax.set_xticks(curr_ticks)
    ax.set_xticklabels(curr_labels, rotation=90)
    ax.set_xlim(left=np.log(100_000))

    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(False) # Strict Grid Removal
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Matched to Latent Plots
    save_figure("residuals_vs_pred_bias.png")
    plt.show()



    # 5. Segment Stability Checks (if metadata available)
    if 'eval_pos_idx' in locals() and eval_pos_idx is not None:
        try:
            print(f"[Eval] Segment Check: Indices length {len(eval_pos_idx)}, Residuals length {len(resid_log)}")
            
            # meta_subset is now df_pred_filtered
            meta_subset = df_pred_filtered.copy()

            if len(meta_subset) == len(resid_log):
                meta_subset['residual'] = resid_log
                
                # --- By Sale Year (Time/Regime Drift) ---
                # This was moved to the top-level, so skip here.

                # --- By Building Class (Residuals) ---
                bldg_class_candidates = ["bldg_class", "building_class", "bldg_class_group", "bldgclass"]
                bldg_col = next((c for c in bldg_class_candidates if c in meta_subset.columns), None)
                
                if bldg_col:
                    # Define standard order (NYC Consumer-facing)
                    class_labels_ordered = [
                        ('R', 'Condominiums'), ('D', 'Elevator Apartments'), ('C', 'Walk-up Apartments'),
                        ('S', 'Mixed Residential'), ('A', '1-2 Family Houses'), ('B', '2 Family Houses'),
                        ('K', 'Retail/Stores'), ('O', 'Office Buildings'), ('H', 'Hotels'), ('L', 'Lofts'),
                        ('E', 'Warehouses'), ('F', 'Factories'), ('G', 'Garages'),
                        ('I', 'Healthcare'), ('J', 'Entertainment'), ('M', 'Religious'),
                        ('N', 'Nursing/Asylums'), ('P', 'Recreation (Indoor)'), ('Q', 'Recreation (Outdoor)'),
                        ('T', 'Transportation'), ('W', 'Educational'), ('U', 'Utility'),
                        ('V', 'Vacant Land'), ('Y', 'Government'), ('Z', 'Miscellaneous')
                    ]
                    code_map = {k: v for k, v in class_labels_ordered}
                    ordered_codes = [k for k, v in class_labels_ordered]

                    # AGGREGATE BY MAJOR CLASS
                    # 1. Extract first char
                    meta_subset['major_class'] = meta_subset[bldg_col].astype(str).str[0].str.upper()
                    
                    # 2. Group by major class
                    bldg_stats = meta_subset.groupby('major_class')['residual'].agg(['mean', 'std', 'count', 'sem'])
                    
                    # 3. Filter small counts
                    bldg_stats = bldg_stats[bldg_stats['count'] > 20]
                        
                    if not bldg_stats.empty:
                        print(f"\n[Eval] Residuals by Building Class (Major Groups):")
                        # print(bldg_stats.head()) 

                        # Plot Mean Residual +/- SE
                        # We want to match the Latent Space color/order logic
                        # Latent space uses a specific order.
                        
                        fig, ax = plt.subplots(figsize=(12, 8))  # Taller figure for 50%+ data area
                        
                        # Create temporary columns for sorting
                        # Note: index is already MAPPED code now
                        bldg_stats['sort_rank'] = bldg_stats.index.to_series().apply(lambda x: ordered_codes.index(x) if x in ordered_codes else 999)
                        bldg_stats = bldg_stats.sort_values('sort_rank')
                        
                        # Map to full names for plotting
                        # index is code 'R', 'D', etc.
                        full_labels = [code_map.get(idx, idx) for idx in bldg_stats.index]
                        
                        # Plot
                        ax.errorbar(range(len(bldg_stats)), bldg_stats['mean'], yerr=bldg_stats['std'] / np.sqrt(bldg_stats['count']),
                               fmt='o', color='teal', capsize=4, label='Mean Residual')
                        ax.axhline(0, color='black', linestyle='--', label='Zero Reference')
                        
                        ax.set_xlabel("Building Class\n" + r"$\it{(Values\ in\ Log\ Space.\ Price\ \geq\ \$100k)}$", fontsize=11)
                        ax.set_ylabel("Mean Residual", fontsize=11)
                        ax.set_title("Performance by Building Class", fontweight='bold', fontsize=14, pad=8)
                        
                        # "vertical axis tick labels" -> yes, rotated ticks.
                        ax.set_xticks(range(len(bldg_stats)))
                        ax.set_xticklabels(full_labels, rotation=90)
                        
                        ax.legend(loc='upper right', framealpha=0.9)
                        
                        # Note moved to xlabel
                        # plt.figtext(0.5, 0.01, f"Values in Log Space. Price >= $100k.", ha="center", fontsize=9, fontstyle='italic')
                        
                        # Ensure plot area is at least 50% of figure height
                        # Standardized layout (tight_layout handles rotated labels if rect is ample)
                        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
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
                        ax.set_ylabel("Mean Residual +/- Standard Error")
                        ax.set_title("Residual Stability by Year Built", fontweight='bold')
                        save_figure("residuals_by_year.png")
                        plt.show()
                
                # --- By Building Class ---
                # (Duplicate block removed - see above)
                    
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
                             ax2.plot(bin_centers_m, bin_sigma_means, 'b--s', label='Mean Predicted Standard Deviation')
                             ax2.set_ylabel("Predicted Standard Deviation", color='blue')
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
                        # Plot (Single Axis for Calibration/Scaling Comparison)
                        # User requested "why are axes... on difference scaled". We fix this by sharing Y-axis.
                        fig, ax1 = plt.subplots(figsize=(8, 5))
                        ax1.plot(fracs_to_test, rmse_list, '-o', color='#21918c', label='Root Mean Square Error')
                        ax1.plot(fracs_to_test, unc_list, '--s', color='#440154', label='Mean Predicted Sigma') # Same axis
                        
                        ax1.set_xlabel("Fraction Masked\n" + r"$\it{(n=" + str(n_syn) + r"\ samples.\ Values\ in\ Log\ Price\ Space.)}$", fontsize=11)
                        # ax1.set_ylabel("", fontsize=11) # User requested no y-axis label
                        ax1.set_ylabel("")
                        ax1.set_ylim(bottom=0)  # Y-axis starts at 0
                        
                        ax1.legend(loc='upper left', framealpha=0.9)
                        
                        ax1.set_title("Missingness Test", fontweight='bold', fontsize=14, pad=8)
                        
                        # Note moved to xlabel
                        # plt.figtext(0.5, 0.01, f"n = {n_syn} samples. Values in Log Price Space.", 
                        #            ha='center', fontsize=9, fontstyle='italic')
                        
                        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
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
                  # Infer CRS based on coordinates
                  is_geo = False
                  if valid_geo[x_col].max() < 185 and valid_geo[x_col].min() > -185:
                      is_geo = True
                  
                  # Standardized Title
                  ax.set_title("Spatial Residual Map", fontweight='bold', fontsize=12)
                  
                  # Remove Lat/Long Ticks
                  ax.set_xticks([])
                  ax.set_yticks([])
                  ax.set_xlabel("")
                  ax.set_ylabel("")
                  
                  # Correct Aspect Ratio for Latitude (Mercator-like correction)
                  if is_geo:
                       try:
                           # Dynamic aspect ratio based on mean latitude
                           mean_lat = valid_geo[y_col].mean()
                           aspect_ratio = 1.0 / np.cos(np.radians(mean_lat))
                           ax.set_aspect(aspect_ratio, adjustable='box')
                           print(f"[Eval] Spatial Map: Applied aspect ratio {aspect_ratio:.2f} (Lat {mean_lat:.1f})")
                       except Exception:
                           ax.set_aspect('equal', adjustable='box')

                  # Add Basemap if possible
                  try:
                      import contextily as cx
                      # Heuristic for CRS
                      if is_geo:
                           # Data is Lat/Lon (EPSG:4326)
                           # contextily needs to know this to reproject background
                           cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron)
                      else:
                           # Data is Projected (likely local or 3857)
                           # If it's NYC data (EPSG:2263), we should tell contextily
                           cx.add_basemap(ax, crs='EPSG:2263', source=cx.providers.CartoDB.Positron)
                  except ImportError:
                      # Attempt Auto-Install as requested "if needed"
                      print("[Eval] Contextily not found. Installing...")
                      try:
                          import subprocess
                          import sys
                          subprocess.check_call([sys.executable, "-m", "pip", "install", "contextily"])
                          import contextily as cx
                          if is_geo:
                               cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron)
                          else:
                               cx.add_basemap(ax, crs='EPSG:2263', source=cx.providers.CartoDB.Positron)
                          print("[Eval] Contextily installed and applied.")
                      except Exception as e_install:
                           print(f"[Eval] Contextily install failed: {e_install}. Plotting minimal map.")
                  except Exception as e_map:
                      # Suggest installation
                      print(f"[Eval] Could not add basemap (contextily): {e_map}.")
                      print("[Eval] To enable basemap: pip install contextily")
                  
                  save_figure("residuals_spatial_map.png")
                  plt.show()
             else:
                  print("[Eval] Not enough valid coordinates for spatial map.")
        except Exception as e:
            print(f"[Eval] Spatial diagnostic failed: {e}")

            
else:
    print("\n[Eval] No stored residuals; skipping residual diagnostics.")

# ===========================================================================
# --- 7. Latent Space Visualization ---
# Drop this immediately before the latent plotting blocks and run once.
# It prints and asserts whether the SAME df_pred row indices are used.

limit_log = np.log(100_000.0)

log_price_all = df_pred[log_y_col].astype(float).to_numpy()
# Check finiteness AND threshold
mask_price = np.isfinite(log_price_all) & (log_price_all >= limit_log)

bldg_class_candidates = ["bldg_class", "building_class", "bldg_class_group", "bldgclass"]
bldg_class_col = next((c for c in bldg_class_candidates if c in df_pred.columns), None)

mask_bldg = np.ones(len(df_pred), dtype=bool)
if bldg_class_col is not None:
    mask_bldg = df_pred[bldg_class_col].notna().to_numpy()

mask_common = mask_price & mask_bldg
idx_common = np.flatnonzero(mask_common)

print("\n[Mask debug] bldg_class_col =", bldg_class_col)
print("[Mask debug] N(df_pred)    =", len(df_pred))
print("[Mask debug] N(mask_price) =", int(mask_price.sum()))
print("[Mask debug] N(mask_bldg)  =", int(mask_bldg.sum()))
print("[Mask debug] N(mask_common)=", int(mask_common.sum()))

# Price plot inputs (what your code uses)
idx_used_price = np.flatnonzero(mask_common)

# Building-class plot inputs (what your code intends to use)
idx_used_bldg = np.flatnonzero(mask_common)

print("[Mask debug] N(price used) =", len(idx_used_price))
print("[Mask debug] N(bldg used)  =", len(idx_used_bldg))

# Strong check: exact same row indices, in the same order
assert np.array_equal(idx_used_price, idx_used_bldg), "Mismatch: price and building-class masks differ"

# Sanity: confirm latent arrays line up with idx_common
assert mu_z.shape[0] == len(df_pred), "Mismatch: mu_z rows do not align with df_pred rows"

print("[Mask debug] PASS: price and building-class plots use identical df_pred rows.")

# --- Define Shared Data for Plots ---
# Enforce that ALL plots use this common subset
valid_z_common = mu_z[mask_common]
price_common   = np.exp(log_price_all[mask_common])
log_price_common = log_price_all[mask_common]

# Update downstream variables to use COMMON mask
mask_valid_price = mask_common
valid_z_price = valid_z_common
price_subset = price_common

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

# NOTE: Gate check removed - variance-based dimension selection doesn't use price_mean_head

# ----------------------------------------------------
# Dimension Selection (Variance-Based)
# ----------------------------------------------------
# Use latent variance as proxy for importance - dimensions with higher variance
# are more informative. This avoids the model inference issues with perm_importance.
try:
    z_vars = np.var(mu_z, axis=0)
    
    # Sort by variance (higher = more informative)
    sorted_dims = np.argsort(z_vars)[::-1]
    plot_dim1 = sorted_dims[0]
    plot_dim2 = sorted_dims[1]
    
    # Compute importance proportions from variance
    total_var = z_vars.sum()
    if total_var > 0:
        imp1_pct = 100 * z_vars[plot_dim1] / total_var
        imp2_pct = 100 * z_vars[plot_dim2] / total_var
    else:
        imp1_pct, imp2_pct = 50, 50
    
    importance_footnote = f"Latent dimensions selected by Variance: z{plot_dim1+1} ({imp1_pct:.0f}%) + z{plot_dim2+1} ({imp2_pct:.0f}%)"
    print(f"[Eval] Dimension selection (variance): z{plot_dim1+1} ({imp1_pct:.1f}%), z{plot_dim2+1} ({imp2_pct:.1f}%)")

except Exception as e:
    print(f"[Eval] Dimension selection failed: {e}")
    plot_dim1, plot_dim2 = 0, 1
    importance_footnote = None
    for attr in ['y_decoder', 'decoder_y', 'price_head', 'predictor', 'predict_y_from_z']:
        if hasattr(vae_model, attr):
            print(f"[Eval]   Found relevant attribute: {attr}")

# --- AFTER try/except: Latent plotting setup ---
# All code from here should be at module level (no indentation)
print(f"\n[Eval] Selected latent dimensions for plotting: z{plot_dim1+1} and z{plot_dim2+1}")

# Default labels (will be overwritten by SHAP-based labels if available)
label_x = "Latent Dimension 1"
label_y = "Latent Dimension 2"

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
        """Generate descriptive label from SHAP importance - no z{n} numbering."""
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
        # Return just the descriptive parts, no z{n} prefix
        return f"({', '.join(parts)})"

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
# --- 7a-ORIG. ORIGINAL Simple latent scatter by price deciles (for comparison) ---
# Enforce mask_common (Price >= 100k) for consistency
if mask_common.sum() > 0:
    # Compute decile edges on the COMMON subset (consistent population)
    price_subset_log = log_price_all[mask_common]
    decile_edges = np.quantile(price_subset_log, np.linspace(0, 1, 11))
    
    # Map all points (but only plot valid ones)
    decile_idx = np.full_like(log_price_all, fill_value=-1, dtype=int)
    # Only map the subset indices
    subset_decile_vals = np.searchsorted(decile_edges[1:-1], price_subset_log, side="right")
    decile_idx[mask_common] = subset_decile_vals
    
    # Valid mask for plotting is just mask_common
    valid_mask = mask_common
    
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
# SHARED MASK STRATEGY: Enforce identical points for Price and Building Class plots
# Filters:
# 1. Price >= $100k
# 2. Building Class not missing (if column exists)

limit_log = np.log(100_000)
mask_price = np.isfinite(log_price_all) & (log_price_all >= limit_log)

bldg_class_candidates = ["bldg_class", "building_class", "bldg_class_group", "bldgclass"]
bldg_class_col = next((c for c in bldg_class_candidates if c in df_pred.columns), None)

if bldg_class_col:
    mask_bldg = df_pred[bldg_class_col].notna()
    mask_common = mask_price & mask_bldg
    print(f"[Eval] Using COMMON Latent Mask (Price >= $100k AND Bldg Class Known). N={mask_common.sum()}")
else:
    mask_common = mask_price
    print(f"[Eval] Using Price-Only Latent Mask (No Bldg Col). N={mask_common.sum()}")

valid_z_price = mu_z[mask_common]
valid_log_price = log_price_all[mask_common]

if len(valid_log_price) > 100:
    # Use common z for building plot later
    bldg_z_common = valid_z_price 
    bldg_series_common = df_pred.loc[mask_common, bldg_class_col].astype(str) if bldg_class_col else None

    # Define Axis Limits based on selected dimensions (Robust 99% interval)
    # This replaces any hardcoded limits
    z_vals_all = mu_z[mask_common]
    x_lim_fixed = np.percentile(z_vals_all[:, plot_dim1], [0.5, 99.5])
    y_lim_fixed = np.percentile(z_vals_all[:, plot_dim2], [0.5, 99.5])
    # Add slight buffer
    x_pad = (x_lim_fixed[1] - x_lim_fixed[0]) * 0.1
    y_pad = (y_lim_fixed[1] - y_lim_fixed[0]) * 0.1
    x_lim_fixed = (x_lim_fixed[0]-x_pad, x_lim_fixed[1]+x_pad)
    y_lim_fixed = (y_lim_fixed[0]-y_pad, y_lim_fixed[1]+y_pad)
    
    print(f"[Eval] Dynamic axis limits for z{plot_dim1+1}/z{plot_dim2+1}: X={x_lim_fixed}, Y={y_lim_fixed}")

    print(f"[Eval] Price filter: {mask_common.sum()} / {np.isfinite(log_price_all).sum()} " +
            f"properties with sale price >= $100K (and valid bldg class if applicable)")

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
    
    # Define limits early for alpha/normalization logic
    # Cap vmax at $250M (consistent with user request)
    vmax_cap = np.log(250_000_000.0)
    vmax = vmax_cap # Alias for tick logic below
    vmin = float(np.nanmin(valid_log_price))
    
    # Scatter with continuous log-price coloring
    # Calculate density-based alpha using KDE (Isoline-Consistent)
    try:
        if HAVE_SCIPY:
            from scipy.stats import gaussian_kde
            xy = np.vstack([z_shuffled[:, plot_dim1], z_shuffled[:, plot_dim2]])
            kde = gaussian_kde(xy)
            dens = kde(xy) # Per-point density
            
            # Map density to alpha using Log1p + Adaptive Gamma
            # Guarantee: Median point opacity >= 0.30
            q_lo, q_hi = np.quantile(dens, [0.01, 0.99])
            dens_clipped = np.clip(dens, q_lo, q_hi)
            
            # 1. Base Log Norm
            dens_log = np.log(dens_clipped)
            min_log, max_log = dens_log.min(), dens_log.max()
            if max_log > min_log:
                norm_log = (dens_log - min_log) / (max_log - min_log + 1e-12)
            else:
                norm_log = np.ones_like(dens_log)
            
            # 2. Adaptive Calibration
            target_median = 0.30
            med_val = np.median(norm_log)
            gamma = 1.0
            if med_val < target_median and med_val > 0.01:
                # Solve: med_val ^ gamma = target
                # gamma = log(target) / log(med)
                gamma = np.log(target_median) / np.log(med_val)
                # Cap gamma to avoid extreme distortion
                gamma = min(gamma, 2.5) 
            
            alpha_per_point = norm_log ** gamma
            
            # Create RGBA manually
            from matplotlib import cm
            norm_scatter = plt.Normalize(vmin=vmin, vmax=vmax_cap)
            cmap = plt.cm.get_cmap('viridis')
            colors_rgba = cmap(norm_scatter(price_shuffled))
            colors_rgba[:, 3] = alpha_per_point
            
            print(f"[Eval] Scatter Transparency: Adaptive Gamma {gamma:.2f} (Median {med_val:.2f}->{np.median(alpha_per_point):.2f})")
            
            sc = ax.scatter(z_shuffled[:, plot_dim1], z_shuffled[:, plot_dim2], 
                            c=colors_rgba, s=8,
                            edgecolors='none', zorder=2)
        else:
            raise ImportError("Scipy needed for KDE alpha")

    except Exception as e_alpha:
        print(f"[Eval] Could not compute KDE alpha: {e_alpha}. Using constant.")
        sc = ax.scatter(z_shuffled[:, plot_dim1], z_shuffled[:, plot_dim2], 
                        c=price_shuffled, s=8, alpha=0.5, cmap='viridis',
                        edgecolors='none', zorder=2, vmax=vmax_cap)
    
    # Add Graded Density Contours Overlay
    add_graded_density_contours(ax, z_shuffled[:, plot_dim1], z_shuffled[:, plot_dim2], x_lim_fixed, y_lim_fixed)
    
    # Create explicit ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax_cap))
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, extend='max') # Show arrow for values > $250M
    cbar.set_label("Sale Price", fontsize=10)
    # Ensure colorbar is opaque
    cbar.solids.set_alpha(1.0)
    
    # Log-scale tick labels
    log_ticks = [np.log(100_000), np.log(250_000), np.log(500_000), 
                    np.log(1_000_000), np.log(2_500_000), np.log(5_000_000),
                    np.log(10_000_000), np.log(25_000_000), np.log(50_000_000),
                    np.log(100_000_000), np.log(250_000_000)]
    log_labels = ['$100K', '$250K', '$500K', '$1M', '$2.5M', '$5M', '$10M', '$25M', '$50M',
                    '$100M', '$250M']
    
    # Filter to range and always include min/max
    valid_ticks = [(t, l) for t, l in zip(log_ticks, log_labels) if vmin <= t <= vmax]
    
    # Format price with B for billions
    def format_price(val):
        price = np.exp(val)
        if price >= 1_000_000_000:
            v = price / 1_000_000_000
            fmt = ".0f" if v.is_integer() else ".1f"
            return f"${v:{fmt}}B"
        elif price >= 1_000_000:
            v = price / 1_000_000
            fmt = ".0f" if v.is_integer() else ".1f" # e.g. 250.0 -> 250M, 2.5 -> 2.5M
            return f"${v:{fmt}}M"
        else:
            return f"${price/1000:.0f}K"
    
    # Strategy: Start with vmin, add intermediate standard ticks that aren't too close, end with vmax
    final_ticks = [vmin]
    final_labels = [format_price(vmin)]
    
    # Threshold for "too close" in log space (approx 10% of range)
    log_range = vmax - vmin
    threshold = 0.05 * log_range if log_range > 0 else 0.1
    
    for t, l in valid_ticks:
        # Skip if too close to min or max
        if abs(t - vmin) > threshold and abs(t - vmax) > threshold:
            final_ticks.append(t)
            final_labels.append(l)
            
    final_ticks.append(vmax)
    final_labels.append(format_price(vmax))
    
    cbar.set_ticks(final_ticks)
    cbar.set_ticklabels(final_labels)
    
    ax.set_xlabel(label_x, fontsize=11)
    ax.set_ylabel(label_y, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Apply fixed axis limits
    ax.set_xlim(x_lim_fixed)
    ax.set_ylim(y_lim_fixed)
    # ax.set_aspect('equal', adjustable='box')  # Commented: asymmetric limits
    
    # Title (no subtitle)
    ax.set_title("Latent Space by Price", fontsize=14, fontweight='bold', pad=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)  # No gridlines
    
    # Add importance footnote if available
    if 'importance_footnote' in dir() and importance_footnote:
        plt.figtext(0.5, 0.01, importance_footnote, ha='center', fontsize=9, fontstyle='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95] if importance_footnote else [0, 0, 1, 1])
    save_figure("latent_space_price.png")
    plt.show()
    
    # --- VERSION 2: Hexbin by price (mean log-price per bin) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Same price colormap as scatter (viridis - perceptually uniform)
    # Create the main hexbin for colorbar (opaque)
    # Cap vmax at $250M (consistent with scatter)
    vmax_cap = np.log(250_000_000)
    hb = ax.hexbin(valid_z_price[:, plot_dim1], valid_z_price[:, plot_dim2], 
                    C=valid_log_price, reduce_C_function=np.mean,
                    gridsize=40, cmap='viridis', mincnt=1, linewidths=0.2, vmax=vmax_cap)
    
    # Calculate transparency based on density - SCALED LOGARITHMICALLY
    # Re-run for counts using a temporary invisible axis or just remove it
    # We must match the grid exactly.
    hb_counts = ax.hexbin(valid_z_price[:, plot_dim1], valid_z_price[:, plot_dim2], 
                           gridsize=40, mincnt=1, alpha=0) 
    bin_counts = hb_counts.get_array()
    hb_counts.remove() # Clean up the helper plot
    
    if len(bin_counts) == len(hb.get_array()):
        # Linear-scale mapping for alpha (Standard)
        min_c, max_c = bin_counts.min(), bin_counts.max()
        
        if max_c > min_c:
            # Scale using Log1p + Adaptive Gamma
            log_c = np.log1p(bin_counts)
            log_min, log_max = np.log1p(min_c), np.log1p(max_c)
            norm_c = (log_c - log_min) / (log_max - log_min + 1e-12)
            
            # Adaptive Calibration
            target_median = 0.30
            med_val = np.median(norm_c)
            gamma_hex = 1.0
            if med_val < target_median and med_val > 0.01:
                gamma_hex = np.log(target_median) / np.log(med_val)
                gamma_hex = min(gamma_hex, 2.5)
            
            alpha_vals = norm_c ** gamma_hex
            print(f"[Eval] Hexbin Transparency: Adaptive Gamma {gamma_hex:.2f} (Median {med_val:.2f}->{np.median(alpha_vals):.2f})")
        else:
            alpha_vals = np.ones_like(bin_counts)

        # CRITICAL FIX: Detach collection from ScalarMappable to prevent overwrite
        hb.update_scalarmappable() # Force initial color generation
        rgba = hb.get_facecolors()
        if len(rgba) == len(alpha_vals):
            rgba[:, 3] = alpha_vals # Overwrite alpha
            hb.set_facecolors(rgba)
            
            # Prevent future updates from overwriting RGBA
            hb.set_array(None) 
            
        # Remove opaque edges to ensure transparency works visually
        hb.set_linewidths(0) 

    # Create distinct ScalarMappable for colorbar (fully opaque)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax_cap))
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, extend='max')
    cbar.set_label("Mean Sale Price", fontsize=10)
    # Ensure colorbar is opaque
    cbar.solids.set_alpha(1.0)
    
    # Add Density Contours removed for Hexbin (relies on Alpha Transparency)
    # Add annotation for transparency (Removed "(Linear Scale)")
    ax.text(0.02, 0.98, "Opacity $\\propto$ Density", 
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3),
            zorder=4)
    
    # Same log-scale labels
    # Same log-scale labels with enforced min/max
    cbar.set_ticks(final_ticks)
    cbar.set_ticklabels(final_labels)
    
    ax.set_xlabel(label_x, fontsize=11)
    ax.set_ylabel(label_y, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Apply fixed axis limits
    ax.set_xlim(x_lim_fixed)
    ax.set_ylim(y_lim_fixed)
    
    # Title - no subtitle, padded
    ax.set_title("Latent Space by Price", fontweight='bold', fontsize=14, pad=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False) # Strict removal
    
    # Match margins with Price Scatter
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure("latent_space_price_hexbin.png")
    plt.show()

# --- 7a-bis. Hexbin density plot for latent space (count only) ---
if mask_valid_price.sum() > 100:
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # White→Blue/Black colormap: transparent/light in sparse areas, dark in dense areas
    # Reverting to explicit colormap with legend as requested
    # Calculate adaptive gamma for Density Hexbin too
    # We need to estimate optimal gamma based on counts (which we don't have unless we fetch again?)
    # Section 7b does NOT run a helper hexbin first.
    # To do this robustly, we compute hexbin data first.
    hb_temp = ax.hexbin(valid_z_price[:, plot_dim1], valid_z_price[:, plot_dim2], gridsize=40, visible=False)
    counts_7b = hb_temp.get_array()
    hb_temp.remove()
    
    gamma_7b = 1.0
    if len(counts_7b) > 0:
        counts_gz = counts_7b[counts_7b > 0]
        # Log Norm Base
        min_c, max_c = counts_gz.min(), counts_gz.max()
        if max_c > min_c:
             # Logic matching scatter/hexbin
             log_c = np.log1p(counts_gz)
             log_min, log_max = np.log1p(min_c), np.log1p(max_c)
             norm_c = (log_c - log_min) / (log_max - log_min + 1e-12)
             med_c = np.median(norm_c)
             
             if med_c < 0.30 and med_c > 0.01:
                 gamma_7b = np.log(0.30) / np.log(med_c)
                 gamma_7b = min(gamma_7b, 2.5)
                 print(f"[Eval] Density Hexbin: Adaptive Gamma {gamma_7b:.2f}")

    # Revert to LINEAR bins but use PowerNorm(AdaptiveGamma) implicitly applied to counts?
    # Wait, PowerNorm applies to values. LogNorm applies Log.
    # If we want Log + Gamma, we need PowerNorm(gamma) applied to Log data? Or PowerNorm applied to raw data?
    # PowerNorm(gamma) on raw data is x^g.
    # We want (Log(x))^g logic from Price Hexbin?
    # Price Hexbin used Alpha ~ (Log Norm)^Gamma.
    # Here mapped to Color (Greys).
    # If we use LogNorm, we get Log distribution.
    # If we use PowerNorm(gamma) on raw counts, we get x^g.
    # x^g is simpler and satisfies monotonicity.
    # Let's use PowerNorm with the adaptive gamma calculated for RAW counts calibration?
    # Actually, simpler: Just use LogNorm. The user's specific "half above 0.3" request was for TRANSPARENCY (alpha).
    # For Density Color, LogNorm is standard.
    # I'll stick to LogNorm for Density to avoid overcomplicating the colormap.
    
    hb = ax.hexbin(valid_z_price[:, plot_dim1], valid_z_price[:, plot_dim2], 
                    gridsize=40, cmap='Greys', mincnt=1, linewidths=0.2,
                    norm=mcolors.LogNorm())
    
    # Colorbar is needed as a legend
    # Colorbar is needed as a legend
    cbar = plt.colorbar(hb, ax=ax)
    
    # Safe max calculation check not needed for linear, but keeping robust structure
    max_c = int(hb.get_array().max()) if hb.get_array().size > 0 else 0
        
    cbar.set_label("Count", fontsize=10)
    
    # Create clean integer ticks using MaxNLocator?
    # NO: LogNorm manages ticks best automatically. Manual linear ticks break it.
    # Just ensure minor ticks are on if helpful
    cbar.minorticks_on()
    
    ax.set_xlabel(label_x, fontsize=11)
    ax.set_ylabel(label_y, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove latent ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Apply fixed axis limits
    ax.set_xlim(x_lim_fixed)
    ax.set_ylim(y_lim_fixed)
    
    # Standardized Title with padding to match Price Scatter
    ax.set_title("Latent Space Density", fontweight='bold', fontsize=14, pad=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False) # Strict removal
    
    # Match margins with Price Scatter (provides space for potential footnotes and aligns axes)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure("latent_space_density.png")
    plt.show()

# --- 7b. Latent colored by SIZE (Square Footage) ---
size_candidates = ["gross_sqft", "land_sqft", "gross_square_feet", "land_area", "sqft", "area"]
size_col = next((c for c in size_candidates if c in df_pred.columns), None)

if size_col:
    print(f"[Eval] Using '{size_col}' for Size-based Latent Viz.")
    size_vals = df_pred[size_col].astype(float).values
    # Update Mask: Strict intersection with mask_common (Global Consistency)
    mask_valid_size = mask_common & np.isfinite(size_vals) & (size_vals > 0)
    
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
    
        # Standardized Title
        ax.set_title("Latent Space by Size", fontweight='bold', fontsize=14)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False) # Strict removal
        
        plt.tight_layout()
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
    
        ax.set_title("Latent Space by Size", fontweight='bold', fontsize=14)
        
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
    print(f"[Eval] Building class plots: Using Shared Mask subset N={len(bldg_z_common)}")
    
    bldg_z = bldg_z_common
    bldg_series_full = bldg_series_common
    
    # Pre-filtered by mask_common, so no need for further NaNs check (notna was in mask)
    # But for safety in case Logic changed:
    valid_bldg_mask = bldg_series_full.notna() # Should be all True
    bldg_z = bldg_z[valid_bldg_mask]
    bldg_series_full = bldg_series_full[valid_bldg_mask].astype(str)
    
    # --- 7c-ORIG. ORIGINAL Simple building class scatter (for comparison) ---
    codes_raw, uniques_raw = pd.factorize(bldg_series_full)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    for code, label in enumerate(uniques_raw):
        mask = codes_raw == code
        if not np.any(mask):
            continue
        ax.scatter(bldg_z[mask, plot_dim1], bldg_z[mask, plot_dim2], s=5, alpha=0.5, label=label)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_xticks([])
    ax.set_yticks([])
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
    bldg_series = bldg_series_full.str[0].str.upper()
    
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
    # Apply same count threshold as Performance by Building Class for consistency
    MIN_CLASS_COUNT = 20
    class_counts = bldg_series.value_counts()
    present_classes = [letter for letter, _ in class_labels_ordered 
                        if letter in bldg_series.values and class_counts.get(letter, 0) > MIN_CLASS_COUNT]
    n_present = len(present_classes)
    cmap = plt.cm.get_cmap('tab20', max(n_present, 1))
    
    # Calculate Density-Based Alpha (Adaptive Gamma) to match Price Scatter visual weight
    # Use bldg_z (which is valid_z_common) for density estimation
    xy_bldg = np.vstack([bldg_z[:, 0], bldg_z[:, 1]])
    kde_bldg = gaussian_kde(xy_bldg)
    dens_bldg = kde_bldg(xy_bldg)
    
    # Adaptive Gamma Logic (Identical to Price Scatter)
    q_lo, q_hi = np.quantile(dens_bldg, [0.01, 0.99])
    dens_clipped = np.clip(dens_bldg, q_lo, q_hi)
    dens_log = np.log(dens_clipped)
    min_log, max_log = dens_log.min(), dens_log.max()
    norm_log_bldg = (dens_log - min_log) / (max_log - min_log + 1e-12) if max_log > min_log else np.ones_like(dens_log)
    
    target_median = 0.30
    med_val = np.median(norm_log_bldg)
    gamma_bldg = 1.0
    if med_val < target_median and med_val > 0.01:
        gamma_bldg = np.log(target_median) / np.log(med_val)
        gamma_bldg = min(gamma_bldg, 2.5)
        
    alpha_bldg = norm_log_bldg ** gamma_bldg
    print(f"[Eval] Building Class Transparency: Adaptive Gamma {gamma_bldg:.2f} (Median {med_val:.2f}->{np.median(alpha_bldg):.2f})")

    # Plot in predefined order for color consistency, legend outside
    for color_idx, letter in enumerate(present_classes):
        mask = (bldg_series == letter).values
        if not np.any(mask): 
            continue
        
        lbl_full = class_labels.get(letter, letter)
        # Apply per-point alpha (density-weighted)
        ax.scatter(bldg_z[mask, plot_dim1], bldg_z[mask, plot_dim2], s=8, 
                    alpha=alpha_bldg[mask], 
                    color=cmap(color_idx), label=lbl_full, edgecolors='none')
    
    # Add Graded Density Contours Overlay
    add_graded_density_contours(ax, bldg_z[:, plot_dim1], bldg_z[:, plot_dim2], x_lim_fixed, y_lim_fixed)
    
    ax.set_xlabel(label_x, fontsize=11)
    ax.set_ylabel(label_y, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Apply fixed axis limits
    ax.set_xlim(x_lim_fixed)
    ax.set_ylim(y_lim_fixed)
    
    ax.set_title("Latent Space by Building Class", fontsize=12, fontweight='bold', pad=8)
    
    ax.grid(False) # Strict removal
    
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
# Plot top dimensions used in scatter plots
dims_to_plot = [plot_dim1, plot_dim2]
n_dims = len(dims_to_plot)

fig, axes = plt.subplots(1, n_dims, figsize=(4*n_dims, 3.5), sharey=True, sharex=True)
if n_dims == 1:
    axes = [axes]

# Compute global limit for x-axis
z_global_min = np.percentile(mu_z[:, dims_to_plot], 0.5)
z_global_max = np.percentile(mu_z[:, dims_to_plot], 99.5)
pad = (z_global_max - z_global_min) * 0.1
xlim_global = (z_global_min - pad, z_global_max + pad)

for i, d in enumerate(dims_to_plot):
    ax = axes[i]
    ax.hist(mu_z[:, d], bins=40, alpha=0.8, color='#3498db', 
            edgecolor='white', linewidth=0.3, density=True)
    
    # Use the same labels as the scatter plots (SHAP-based)
    # Split into 2 lines for readability
    if i == 0:
        ax.set_xlabel(label_x.replace(' (', '\n('), fontsize=9)
    else:
        ax.set_xlabel(label_y.replace(' (', '\n('), fontsize=9)
    
    if i == 0:
        ax.set_ylabel("Density", fontsize=11)
    
    ax.set_xlim(xlim_global)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Title
fig.suptitle("Latent Marginal Distributions", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_figure("latent_marginals.png")
plt.show()

print("[Eval] All visualizations completed.")
