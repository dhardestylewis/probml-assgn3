# probml.evaluation

Post-training evaluation, visualization, and validation utilities for SemiSupMIWAE.

## Main Entry Points

### `run_full_evaluation(run_dir, output_dir=None)`

Complete post-training evaluation pipeline including:
- Posterior predictive metrics (NLL, RMSE, MAE, MAPE)
- Residual diagnostics (histogram, QQ-plot)  
- Latent space visualization (z1-z2 scatter by price decile / building class)
- Residual structure analysis (R² of residuals given X and z)

```python
from probml.evaluation import run_full_evaluation

# Full evaluation with plots saved to disk
results = run_full_evaluation(
    run_dir="results/2025-01-01_run/",
    output_dir="plots/",
    show_plots=True
)

print(f"RMSE (log-price): {results['metrics_cv']['rmse_log']:.4f}")
print(f"R²(residual | z): {results['residual_structure']['r2_residual_given_z']:.4f}")
```

### `run_convergence_analysis(results_pkl_path, save_plot_path=None)`

Plot training/validation loss curves across folds.

```python
from probml.evaluation import run_convergence_analysis

result = run_convergence_analysis(
    "results/miwae_analysis_results.pkl",
    save_plot_path="plots/convergence.png"
)
print(f"Mean epochs: {np.mean(result['epochs_per_fold']):.1f}")
```

## Command Line Usage

```bash
# Convergence analysis
python -m probml.evaluation.convergence results/miwae_analysis_results.pkl

# Full post-training evaluation  
python -m probml.evaluation.post_training results/ plots/
```

## Module Contents

### `convergence.py`
| Function | Description |
|----------|-------------|
| `run_convergence_analysis()` | Main entry point |
| `plot_convergence()` | Plot train/val loss curves |
| `extract_fold_histories()` | Parse history from various formats |
| `summarize_epochs_per_fold()` | Get epochs trained per fold |

### `post_training.py`
| Function | Description |
|----------|-------------|
| `run_full_evaluation()` | Main entry point |
| `load_results()` | Load results_summary and predictions |
| `compute_metrics()` | Calculate all posterior predictive metrics |
| `plot_residual_histogram()` | Residual distribution plot |
| `plot_residual_qq()` | QQ-plot vs Normal |
| `plot_latent_by_price_decile()` | z1-z2 scatter by price |
| `plot_latent_by_building_class()` | z1-z2 scatter by building class |
| `plot_latent_marginals()` | Marginal histograms of latent dims |
| `analyze_residual_structure()` | R² analysis of residuals |

### `validation.py`
Cross-validation utilities and metric computation.

### `latent_eda.py`
Extended latent space exploratory data analysis.
