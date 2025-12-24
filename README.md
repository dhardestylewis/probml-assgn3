# probml-assgn3

Probabilistic Machine Learning Assignment 3: Semi-Supervised MIWAE for NYC Real Estate Price Prediction

## Project Structure

```
probml-assgn3/
├── code/
│   ├── probml/                    # Modular package (NEW)
│   │   ├── core/                  # Logging, ICA utilities
│   │   ├── preprocessing/         # DataPreprocessor, SamplingUtils
│   │   ├── analysis/              # FeatureSelector, Clustering, Orchestrator
│   │   ├── models/                # VAE, MIWAE, Trainer, Student-t
│   │   └── evaluation/            # Convergence, post-training eval
│   ├── retabularautovae_probml.py # Original monolith (9206 lines)
│   └── post_training_eval.py      # Post-training evaluation (2669 lines)
├── final_project/                 # LaTeX report and poster
└── references/                    # Papers and submissions
```

## Quick Start

```bash
# Run training pipeline
cd code
python -m probml.main_pipeline

# Run post-training evaluation
python post_training_eval.py
```

## Package Usage

```python
from probml.core import get_logger
from probml.models import SemiSupMIWAE, VAETrainer
from probml.evaluation import run_convergence_analysis

# Train model
model = SemiSupMIWAE(input_dim=50, latent_dim=5, y_dim=1)
trainer = VAETrainer(model)
trainer.train(train_data, val_data, epochs=100)

# Evaluate
run_convergence_analysis("results/miwae_analysis_results.pkl")
```

## Key Components

| Module | Description |
|--------|-------------|
| `SemiSupMIWAE` | Semi-supervised VAE with Student-t mixture prior |
| `VAETrainer` | Training loop with KLD annealing, early stopping |
| `FeatureSelector` | Multi-method feature selection |
| `AnalysisOrchestrator` | Full preprocessing + analysis pipeline |

## References

- Course: Columbia STCS 6701 (Prof. David Blei)
- Data: NYC Rolling Sales
