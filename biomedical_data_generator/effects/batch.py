# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Batch effect utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_batches(
    n_samples: int,
    n_batches: int,
    *,
    rng: np.random.Generator,
    y: NDArray[np.int64] | None = None,
    confounding: float = 0.0,  # in [-1, 1]; 0 = unabhängig
) -> NDArray[np.int64]:
    """Sample per-sample batch ids in {0, ..., n_batches-1}.

    If `confounding != 0` and `y` is provided, classes are softly aligned to different batches.
    Positive values bias classes towards their "preferred" batch; negative values invert the bias.
    """
    if n_batches <= 1:
        return np.zeros(n_samples, dtype=np.int64)

    if y is None or abs(confounding) < 1e-12:
        return rng.integers(0, n_batches, size=n_samples, dtype=np.int64)

    # simple cyclic class→batch preference
    K = int(y.max()) + 1
    prefs = np.arange(n_batches) % max(K, 1)

    # logits per sample over batches
    logits = np.zeros((n_samples, n_batches), dtype=float)
    for k in range(K):
        mask = y == k
        logits[mask, :] = -1.0
        logits[mask, prefs == k] = 1.0

    # scale by confounding
    logits *= 3.0 * float(confounding)

    # softmax sampling
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)

    cdf = P.cumsum(axis=1)
    r = rng.random(size=(n_samples, 1))
    return (r > cdf[:, :-1]).sum(axis=1).astype(np.int64)


def add_batch_intercepts(
    X: NDArray[np.float64],
    batch: NDArray[np.int64],
    *,
    sigma: float = 0.5,
    cols: NDArray[np.int64] | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Add random intercepts b_g ~ N(0, sigma^2) per batch g to selected columns in-place.

    X[i, j] += b_{batch[i]}  for j in `cols` (or all columns if cols is None).
    """
    if X.size == 0 or batch.size == 0:
        return
    if cols is None:
        cols = np.arange(X.shape[1], dtype=np.int64)
    rng = np.random.default_rng() if rng is None else rng
    n_batches = int(batch.max()) + 1
    b = rng.normal(loc=0.0, scale=float(sigma), size=n_batches)
    X[:, cols] += b[batch][:, None]


# import numpy as np
# from numpy._typing import NDArray
#
# from ..config import BatchEffectsConfig
# from ..meta import BatchMeta
#
# def apply_batch_effects(
#     X: np.ndarray,
#     y: np.ndarray,
#     affected_cols: np.ndarray,
#     cfg: BatchEffectsConfig,
#     rng: np.random.Generator,
# ) -> tuple[np.ndarray, BatchMeta]:
#     n, _ = X.shape
#     G = int(cfg.n_batches)
#     offsets = rng.normal(0.0, float(cfg.sd), size=G)
#
#     if not cfg.confounded:
#         batch_ids = rng.integers(0, G, size=n, dtype=np.int32)
#         majors = None
#     else:
#         majors = np.zeros(G, dtype=np.int32)
#         majors[: G // 2] = 1
#         rng.shuffle(majors)
#         idx0 = np.where(majors == 0)[0]; idx1 = np.where(majors == 1)[0]
#         p = float(cfg.p_major)
#         batch_ids = np.empty(n, dtype=np.int32)
#         for i in range(n):
#             pick = idx1 if (y[i] == 1 and rng.random() < p) else idx0 if (y[i] == 0 and rng.random() < p)
#             else (idx0 if y[i]==1 else idx1)
#             batch_ids[i] = int(rng.choice(pick))
#
#     X_out = X.copy()
#     X_out[np.arange(n)[:, None], affected_cols[None, :]] += offsets[batch_ids][:, None]
#
#     meta = BatchMeta(
#         batch_ids=batch_ids,
#         batch_offsets=offsets,
#         batches_majority_class=majors,
#         scope=cfg.scope,
#         sd=cfg.sd,
#     )
#     return X_out, meta
#
#
# def add_batch_intercepts(
#     X: NDArray[np.float64],
#     batch: NDArray[np.int64],
#     *,
#     sigma: float = 0.5,
#     cols: NDArray[np.int64] | None = None,
#     rng: np.random.Generator | None = None,
# ) -> None:
#     """Additive random intercepts per batch: X[i, j] += b_{batch[i]} for j in cols."""
#     if X.size == 0 or batch.size == 0:
#         return
#     if cols is None:
#         cols = np.arange(X.shape[1], dtype=np.int64)
#     rng = np.random.default_rng() if rng is None else rng
#     n_batches = int(batch.max()) + 1
#     b = rng.normal(loc=0.0, scale=sigma, size=n_batches)
#     X[:, cols] += b[batch][:, None]
#
#
# def make_batches(
#     n_samples: int,
#     n_batches: int,
#     *,
#     rng: np.random.Generator,
#     y: NDArray[np.int64] | None = None,
#     confounding: float = 0.0,  # in [-1, 1], 0 = unabhängig
# ) -> NDArray[np.int64]:
#     """Sample batch IDs; bei confounding>0 werden Klassen preferenziell in bestimmte Batches gelegt."""
#     if n_batches <= 1:
#         return np.zeros(n_samples, dtype=np.int64)
#     if y is None or abs(confounding) < 1e-12:
#         return rng.integers(0, n_batches, size=n_samples, dtype=np.int64)
#
#     # simpel: jeder Klasse ein "Lieblingsbatch", weiche Zuordnung über Softmax
#     K = int(y.max()) + 1
#     prefs = np.arange(n_batches) % K  # zyklische Zuordnung von Batches zu Klassen
#     logits = np.full((n_samples, n_batches), 0.0)
#     for k in range(K):
#         mask = (y == k)
#         logits[mask, :] = -1.0
#         logits[mask, prefs == k] = 1.0
#     logits *= 3.0 * float(confounding)  # Skala
#     P = np.exp(logits - logits.max(axis=1, keepdims=True))
#     P /= P.sum(axis=1, keepdims=True)
#     cdf = P.cumsum(axis=1)
#     r = rng.random(size=(n_samples, 1))
#     return (r > cdf[:, :-1]).sum(axis=1).astype(np.int64)
#
