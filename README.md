# ActivationRank

ActivationRank is a small Python/NumPy library for analyzing two-dimensional activation matrices (samples × features). It provides low-rank decompositions, random projection, reconstruction and retained-variance metrics, and column-level variance/error rankings. The implementation is a focused numerical utility, not a model-instrumentation or activation-capture framework.

## Status

Version 0.1.0. The repository contains a single implementation module and a focused pytest suite. Core matrix operations are implemented, but integration with machine-learning frameworks and evaluation of downstream model behavior are outside the current code. No benchmark or downstream-accuracy results are provided here.

## Requirements and installation

The project metadata requires Python 3.9 or later and NumPy 1.24 or later. Install from the repository root with:

```bash
python -m pip install .
```

To install the test dependencies and run the suite:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Usage

Functions accept NumPy-compatible, two-dimensional matrices. Rows represent samples and columns represent features.

```python
import numpy as np
from activationrank import reconstruction_error, retained_variance, svd_compress

A = np.array([
    [1.0, 2.0, 0.0],
    [2.0, 4.0, 1.0],
    [3.0, 6.0, 0.0],
])

# Compute a full set of singular values, then form a rank-2 approximation.
u, s, vt = svd_compress(A, rank=min(A.shape))
k = 2
approx = (u[:, :k] * s[:k]) @ vt[:k, :]
print("relative error:", reconstruction_error(A, approx))
print("retained variance:", retained_variance(s, k))
```

`svd_compress` returns `(u, s, vt)` truncated to the requested rank (capped at the matrix dimensions). To calculate retained variance relative to the full spectrum, pass enough rank to retain all singular values, as in the example.

## Implemented functions

- `ActivationStore`: append batches under a layer name, retrieve their concatenated matrix with `matrix(layer)`, and list known names with `layers()`.
- `svd_compress(matrix, rank=8)`: truncated SVD factors.
- `pca_compress(matrix, components=8)`: PCA by covariance eigendecomposition; returns `(scores, mean, components)`. Reconstruct an approximation with `scores @ components.T + mean` when using all returned components.
- `random_project(matrix, dim=16, seed=0)`: Gaussian random projection; returns projected values and the projection matrix. It does not provide an inverse or reconstruction routine.
- `reconstruction_error(matrix, approx)`: relative Frobenius-norm error, requiring matching shapes.
- `retained_variance(s, rank)`: fraction of squared singular-value magnitude in the first `rank` values supplied.
- `column_variance(matrix)` and `rank_columns(matrix, top=0)`: per-column variances and descending variance ranking.
- `reconstruction_impact(matrix, rank=8)`: each column's share of squared error under a rank-limited SVD reconstruction.

## Tests and CI

On the reviewed repository commit, the local test command `python3 -m pytest -q -p no:cacheprovider` completed with **12 passed**. A separate install and import smoke check in a disposable copy also succeeded. These are local checks, not a report of a remote GitHub Actions run.

The CI workflow is configured to run pytest, compile the module and tests, and perform a small SVD smoke check on Python 3.10, 3.11, and 3.12. The project metadata declares Python 3.9+, but the configured CI matrix does not include 3.9. The checked-in tests do not directly exercise `column_variance`, `rank_columns`, or `reconstruction_impact`. A CodeQL workflow is also configured; no remote CodeQL result is claimed here.

## Scope and limitations

- This library operates on matrices supplied by the caller. It does not install model hooks, collect live activations from a framework, or provide a training/inference pipeline.
- Projection and factorization utilities provide numerical summaries, not evidence that compressed activations preserve downstream task behavior.
- Random projection returns a lower-dimensional projection, not a reconstructed approximation; reconstruction error requires a same-shaped approximation supplied by the caller.
- `ActivationStore` appends batches and concatenates them on retrieval; it does not validate matching feature widths when batches are added.
- No performance benchmark, model-behavior study, or production-readiness claim is included.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for its terms and copyright notice (Kapildhinwa2010-dev, 2026).
