# probml.analysis

Feature selection, clustering, and dimensionality reduction modules.

## Modules

### `feature_selection.py`
- `FeatureSelectorConfig`: Configuration dataclass
- `FeatureSelector`: Multi-method feature selection (variance, Laplacian score, pseudo-labeling)

### `dimensionality.py`
- `DimensionalityReducer`: Intrinsic dimension estimation (TwoNN, Levina-Bickel) and ICA

### `clustering.py`
- `ClusteringSuite`: KMeans, GMM, HDBSCAN with automatic cluster count selection

### `spatial.py`
- `SpatialAnalyzer`: Geospatial analysis using coordinates (Moran's I, LISA)

### `orchestrator.py`
- `AnalysisOrchestrator`: Coordinates full preprocessing + analysis pipeline

## Usage

```python
from probml.analysis.feature_selection import FeatureSelector, FeatureSelectorConfig
from probml.analysis.orchestrator import AnalysisOrchestrator

# Feature selection
selector = FeatureSelector(df, target_col='sale_price')
top_features = selector.select(auto_k_selection_method='coverage')

# Full pipeline
orchestrator = AnalysisOrchestrator(price_col='sale_price')
results = orchestrator.full_analysis_pipeline(df, numeric_features=top_features)
```
