# probml

A modular Python package for probabilistic machine learning on real estate tabular data, featuring Semi-Supervised MIWAE (Missing Importance Weighted Autoencoder) with Student-t mixture priors.

## Package Structure

```
probml/
├── core/           # Core utilities (logging, ICA)
├── preprocessing/  # Data cleaning, transformation, sampling
├── analysis/       # Feature selection, clustering, dimensionality reduction
├── models/         # VAE architectures, trainers, distributions
├── evaluation/     # Validation, latent space EDA, convergence plotting
├── data_loading.py # NYC real estate data loading utilities
└── main_pipeline.py # Main training pipeline entry point
```

## Quick Start

### Running the Training Pipeline

```bash
cd code
python -m probml.main_pipeline
```

Or import and run programmatically:

```python
from probml.models.trainer import VAETrainer
from probml.models.miwae import SemiSupMIWAE
from probml.models.vae_helpers import create_vae_from_artifacts, prepare_vae_input
```

### Running Post-Training Evaluation

```python
from probml.evaluation.convergence import run_convergence_analysis
from probml.evaluation.post_training import run_evaluation

# Plot convergence
run_convergence_analysis("results/miwae_analysis_results.pkl")

# Full evaluation with metrics and plots
run_evaluation("results/", output_dir="plots/")
```

## Subpackage Descriptions

| Subpackage | Description |
|------------|-------------|
| `core` | Logging utilities (`get_logger`), library availability checks, ICA wrapper |
| `preprocessing` | `DataPreprocessor` for cleaning/scaling, `SamplingUtils` for balanced sampling |
| `analysis` | `FeatureSelector`, `ClusteringSuite`, `DimensionalityReducer`, `AnalysisOrchestrator` |
| `models` | `VariationalAutoencoderBase`, `SemiSupMIWAE`, `VAETrainer`, Student-t distributions |
| `evaluation` | Convergence plots, residual diagnostics, latent space visualization |

## Entry Points

1. **Training**: `python -m probml.main_pipeline` (full 10-fold CV training)
2. **Convergence Analysis**: `python -m probml.evaluation.convergence <results.pkl>`
3. **Post-Training Eval**: `python -m probml.evaluation.post_training <results_dir/>`

## Dependencies

- PyTorch >= 1.10
- scikit-learn >= 1.0
- pandas, numpy, matplotlib
- Optional: faiss-cpu/gpu, picard (for advanced ICA), hdbscan, skdim
