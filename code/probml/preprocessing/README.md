# probml.preprocessing

Data preprocessing and sampling utilities.

## Modules

### `data_preprocessor.py`
`DataPreprocessor` class handles:
- Price column cleaning and log1p transformation
- Outlier detection (IsolationForest, percentile-based)
- BBL identifier creation
- Skewed feature transformation (quantile, Yeo-Johnson, Box-Cox)
- StandardScaler application

```python
from probml.preprocessing.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(
    price_col='sale_price',
    outlier_method='isolation_forest',
    skewness_threshold=1.0
)
df_processed = preprocessor.fit_transform(df)
```

### `sampling.py`
`SamplingUtils` class provides:
- Balanced/stratified sampling
- Price-band stratification
- Sample weighting for imbalanced data

```python
from probml.preprocessing.sampling import SamplingUtils

sampler = SamplingUtils(random_state=42)
df_balanced = sampler.balanced_sample(df, target_col='log_sale_price', n_samples=10000)
```
