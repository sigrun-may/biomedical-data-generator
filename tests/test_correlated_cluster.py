# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for the correlated cluster generator."""

from typing import Literal

import numpy as np
import pytest

from biomedical_data_generator.generator import generate_correlated_cluster

Structure = Literal["equicorrelated", "toeplitz"]
STRUCTURES: tuple[Structure, ...] = ("equicorrelated", "toeplitz")


def _mean_offdiag(corr: np.ndarray) -> float:
    d = np.eye(corr.shape[0], dtype=bool)
    off = corr[~d]
    return float(off.mean()) if off.size else 1.0


@pytest.mark.parametrize("structure", STRUCTURES)
def test_generate_correlated_cluster_shapes_and_correlation(structure: Structure) -> None:
    X, meta = generate_correlated_cluster(n_samples=200, size=5, rho=0.6, structure=structure, random_state=123)
    assert X.shape == (200, 5)
    assert "corr_matrix" in meta and isinstance(meta["corr_matrix"], np.ndarray)
    C = meta["corr_matrix"]
    assert C.shape == (5, 5)
    # diagonal ~ 1
    assert np.allclose(np.diag(C), 1.0, atol=1e-6)
    # mean off-diagonal roughly near rho (for equicorrelated) or somewhat lower (toeplitz)
    m = _mean_offdiag(C)
    assert 0.3 <= m <= 0.9  # broad sanity window


def test_toeplitz_accepts_negative_rho():
    # AR(1)/Toeplitz is defined for |rho|<1 even if rho < 0
    X, meta = generate_correlated_cluster(n_samples=150, size=6, rho=-0.4, structure="toeplitz", random_state=1)
    C = meta["corr_matrix"]
    assert C.shape == (6, 6)
    # alternating signs across distance: first off-diagonal should be ~rho
    assert C[0, 1] < 0 and abs(C[0, 1]) > 0.2
