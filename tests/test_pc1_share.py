# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np
import pandas as pd
import pytest

from biomedical_data_generator.utils.correlation_tools import pc1_share, pc1_share_from_corr, variance_partition_pc1


def _equicorr_matrix(p: int, correlation: float) -> np.ndarray:
    """Equicorrelation matrix with 1 on diag and correlation off-diagonal."""
    if not (0 <= p):
        raise ValueError("p must be non-negative")
    if p == 0:
        return np.zeros((0, 0), dtype=float)
    M = np.full((p, p), correlation, dtype=float)
    np.fill_diagonal(M, 1.0)
    return M


@pytest.mark.parametrize("p", [1, 2, 5, 10])
def test_pc1_share_from_corr_identity(p: int):
    C = np.eye(p, dtype=float)
    val = pc1_share_from_corr(C)
    # For identity: largest eigenvalue = 1, trace = p -> EVR = 1/p
    expected = 1.0 / max(p, 1)
    assert np.isfinite(val)
    assert 0.0 <= val <= 1.0
    assert np.isclose(val, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("p,correlation", [(3, 0.2), (6, 0.7), (8, 0.4)])
def test_pc1_share_from_corr_equicorr_formula(p: int, correlation: float):
    C = _equicorr_matrix(p, correlation)
    # Equicorr eigenvalues: lambda_max = 1 + (p-1)correlation; others = 1 - correlation
    expected = (1.0 + (p - 1) * correlation) / p
    got = pc1_share_from_corr(C)
    assert np.isfinite(got)
    assert 0.0 <= got <= 1.0
    assert np.isclose(got, expected, rtol=0, atol=1e-12)


def _make_shared_signal_block(n: int, p: int, correlation_target: float, seed: int = 0) -> pd.DataFrame:
    """Generate correlated features via a shared latent z (anchor + proxies)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    # choose sigma^2 so that corr ≈ a^2/(a^2+sigma^2) with a=1
    sigma2 = (1.0 - correlation_target) / max(correlation_target, 1e-9)
    sigma = float(np.sqrt(max(sigma2, 1e-12)))
    X = np.column_stack([z + rng.normal(scale=sigma, size=n) for _ in range(p)]).astype(float)
    # standardize columns
    X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-9)
    cols = ["anchor"] + [f"proxy_{j}" for j in range(1, p)]
    return pd.DataFrame(X, columns=cols)


def test_pc1_share_df_vs_corr_close():
    n, p, correlation = 400, 6, 0.6
    X_df = _make_shared_signal_block(n, p, correlation, seed=1)
    C = X_df.corr(method="pearson").to_numpy(dtype=float)
    s1 = pc1_share(X_df, method="pearson", rowvar=False)
    s2 = pc1_share_from_corr(C)
    # Finite-sample: allow small deviation
    assert 0.0 <= s1 <= 1.0 and 0.0 <= s2 <= 1.0
    assert np.isclose(s1, s2, rtol=0, atol=5e-3)


def test_pc1_share_numpy_rowvar_equivalence():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(200, 5)).astype(float)  # (n_samples, n_features)
    s_col = pc1_share(X, method="pearson", rowvar=False)
    s_row = pc1_share(X.T, method="pearson", rowvar=True)
    assert np.isclose(s_col, s_row, rtol=0, atol=1e-12)


def test_variance_partition_matches_pc1_and_counts():
    X_df = _make_shared_signal_block(n=300, p=7, correlation_target=0.5, seed=7)
    vp = variance_partition_pc1(X_df, method="pearson", rowvar=False)
    s = pc1_share(X_df, method="pearson", rowvar=False)
    assert set(vp.keys()) == {"n_features", "pc1_evr", "pc1_var_ratio"}
    assert vp["n_features"] == X_df.shape[1]
    assert np.isclose(vp["pc1_evr"], s, rtol=0, atol=1e-12)
    assert np.isclose(vp["pc1_var_ratio"], s, rtol=0, atol=1e-12)
    assert 0.0 <= vp["pc1_evr"] <= 1.0


def test_spearman_path_runs_and_bounds():
    X_df = _make_shared_signal_block(n=250, p=5, correlation_target=0.4, seed=5)
    # apply a monotone transform to check robustness of the code path
    X_mon = X_df.apply(np.exp)
    val = pc1_share(X_mon, method="spearman", rowvar=False)
    assert np.isfinite(val)
    assert 0.0 <= val <= 1.0


def test_input_validation_errors():
    with pytest.raises(ValueError):
        pc1_share_from_corr(np.ones((3, 4)))  # not square

    with pytest.raises(ValueError):
        pc1_share(np.ones(10))  # not 2D


def test_pc1_share_from_corr_empty_matrix():
    """Test pc1_share_from_corr with empty matrix."""
    C = np.zeros((0, 0), dtype=float)
    result = pc1_share_from_corr(C)
    assert result == 0.0


def test_pc1_share_invalid_method_dataframe():
    """Test pc1_share with invalid method for DataFrame."""
    df = pd.DataFrame(np.random.randn(10, 3))
    with pytest.raises(ValueError, match="method must be"):
        pc1_share(df, method="invalid_method")


def test_pc1_share_spearman_numpy():
    """Test pc1_share with spearman method on numpy array."""
    X = np.random.randn(50, 4)
    result = pc1_share(X, method="spearman", rowvar=False)
    assert 0.0 <= result <= 1.0
    assert np.isfinite(result)


def test_pc1_share_kendall_numpy():
    """Test pc1_share with kendall method on numpy array (should raise error)."""
    X = np.random.randn(50, 4)
    with pytest.raises(ValueError, match="method must be"):
        pc1_share(X, method="kendall", rowvar=False)
