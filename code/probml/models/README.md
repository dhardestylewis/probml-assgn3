# probml.models

VAE architectures, training utilities, and distributions.

## Modules

### `vae_base.py`
- `VariationalAutoencoderBase`: Base VAE with configurable encoder/decoder, Student-t mixture prior

### `miwae.py`
- `SemiSupMIWAE`: Semi-supervised MIWAE extending base VAE with price prediction head

### `trainer.py`
- `VAETrainer`: Training loop with KLD annealing, early stopping, imputation, and model I/O

### `vae_helpers.py`
- `prepare_vae_input()`: Convert DataFrame to tensors with masks
- `create_vae_from_artifacts()`: Instantiate model from orchestrator artifacts

### `distributions.py`
- `StudentTDistribution`: Multivariate Student-t for heavy-tailed priors
- `MixtureDistribution`: Mixture of distributions

### `student_t.py`
- Student-t mixture model utilities for clustering prior

## Usage

```python
from probml.models.miwae import SemiSupMIWAE
from probml.models.trainer import VAETrainer
from probml.models.vae_helpers import prepare_vae_input

# Prepare data
X, X_mask, y, y_mask, feature_names, _, _ = prepare_vae_input(df, feature_cols, target_col)

# Create model
model = SemiSupMIWAE(input_dim=X.shape[1], latent_dim=5, y_dim=1)

# Train
trainer = VAETrainer(model, learning_rate=3e-4)
history = trainer.train(train_dataset, val_dataset, epochs=100)
```
