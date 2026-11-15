# ActivationRank

[![CI](https://github.com/kapildhinwa2010-dev/ActivationRank/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ActivationRank/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/ActivationRank/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ActivationRank/actions/workflows/codeql.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Rank-based activation compression: truncated SVD, PCA, random projections with error/variance scoring.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

Compress activation matrices three ways - truncated SVD, PCA, random
projection - and score the result with relative reconstruction error
and retained variance. ActivationStore collects per-layer batches.

## Install

    pip install -r requirements.txt

## Quick start

    from activationrank import svd_compress, retained_variance

    u, s, vt = svd_compress(A, rank=4)
    print(retained_variance(s, 4))

## API

- svd_compress(A, rank), pca_compress(A, components),
  random_project(A, dim, seed)
- reconstruction_error(A, approx), retained_variance(s, rank)
- ActivationStore: add / matrix / layers

## Honest results and limitations

- Compression here means matrix factorization of collected activations,
  not runtime hook compression (see ActivationCompress for hooks).
- No downstream-accuracy evaluation: retained variance is a proxy, and
  the source experiments did not show it guarantees task behaviour.
- Random projection errors depend on spectrum and dim; PCA returns
  centered scores (add the mean back before comparing).

## License

MIT - see [LICENSE](LICENSE).
