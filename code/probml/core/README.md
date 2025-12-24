# probml.core

Core utilities and foundational modules.

## Modules

### `utils.py`
- `get_logger(name, verbose)`: Creates a configured logger instance
- `check_available_methods()`: Checks for optional library availability (FAISS, Picard, HDBSCAN, skdim)
- Exports: `FAISS_AVAILABLE`, `PICARD_AVAILABLE`, `HDBSCAN_AVAILABLE`, `SKDIM_AVAILABLE`

### `ica.py`
- `PicardResult`: NamedTuple for Picard ICA results (compatible with sklearn FastICA interface)
- `ICAModule`: Wrapper that attempts Picard first, falls back to FastICA

## Usage

```python
from probml.core.utils import get_logger, FAISS_AVAILABLE
from probml.core.ica import ICAModule

logger = get_logger("MyModule", verbose=True)
logger.info("Starting analysis...")

if FAISS_AVAILABLE:
    # Use fast k-NN
    pass
```
