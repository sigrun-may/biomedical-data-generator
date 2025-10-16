# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Minimal, readable class-wise shifting with two knobs."""

from collections.abc import Iterable

import numpy as np


def shift_classes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    informative_idx: Iterable[int],
    anchor_contrib: dict[int, tuple[float, int]] | None = None,  # col -> (beta, cls)
    class_sep: float = 1.0,
    anchor_strength: float = 1.0,
    anchor_mode: str = "equalized",  # "equalized" or "strong"
    spread_non_anchors: bool = True,
) -> None:
    """Minimal, readable class-wise shifting with two knobs.

    Args:
        X: Feature matrix (modified in place).
        y: Class labels in {0, ..., K-1}.
        informative_idx: indices of informative (non-anchor) features.
        anchor_contrib: optional mapping col -> (beta, cls)
            - beta scales the anchor contribution per feature.
            - cls is the target class for the one-vs-rest anchor.
        class_sep: global separation scale for non-anchors (spread across classes).
        anchor_strength: additional scale for anchors (one-vs-rest).
        anchor_mode: "equalized" (default) or "strong".
            - "equalized": anchor strength is independent of K (recommended).
            - "strong": anchor strength grows with K (can overwhelm non-anchors).
        spread_non_anchors: if True (default), spread non-anchor informative features across classes.
    """
    if X.size == 0:
        return
    K = int(np.max(y)) + 1
    if K <= 1:
        return

    # 1) Spread for non-anchors
    if spread_non_anchors and K > 1:
        denom = K - 1
        spread_vec = class_sep * (np.arange(K, dtype=float) - (K - 1) / 2) / denom
        anchor_cols = set(anchor_contrib.keys()) if anchor_contrib else set()
        for idx in informative_idx:
            if idx in anchor_cols:
                continue
            for k in range(K):
                X[y == k, idx] += spread_vec[k]

    # 2) Anchors: one-vs-rest
    if anchor_contrib:
        for col, (beta, cls) in anchor_contrib.items():
            if anchor_mode == "equalized":
                A = class_sep * anchor_strength * float(beta) * (K - 1) / K
            elif anchor_mode == "strong":
                A = class_sep * anchor_strength * float(beta) * (K - 1) / 2
            else:
                raise ValueError(f"Unknown anchor_mode={anchor_mode!r}")
            X[y == cls, col] += A
            for k in range(K):
                if k != cls:
                    X[y == k, col] -= A / (K - 1)
    # # Determine number of classes
    # K = int(np.max(y)) + 1 if X.shape[0] > 0 else 0
    # if K <= 1 or class_sep == 0.0:
    #     return
    #
    # # 1) Spread for non-anchors (optional)
    # if spread_non_anchors and K > 1:
    #     denom = K - 1
    #     spread_vec = class_sep * (np.arange(K, dtype=float) - (K - 1) / 2) / denom
    #     anchor_cols = set(anchor_contrib.keys()) if anchor_contrib else set()
    #
    #     for idx in informative_idx:
    #         if idx in anchor_cols:
    #             continue
    #         for k in range(K):
    #             X[y == k, idx] += spread_vec[k]
    #
    # # 2) Anchors (one-vs-rest), K-invariant strength:
    # #    A = class_sep * anchor_strength * beta * (K-1) / K
    # if anchor_contrib:
    #     for col, (beta, cls) in anchor_contrib.items():
    #         A = class_sep * anchor_strength * float(beta) * (K - 1) / K
    #         X[y == cls, col] += A
    #         for k in range(K):
    #             if k != cls:
    #                 X[y == k, col] -= A / (K - 1)
