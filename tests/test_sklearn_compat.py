import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from sklearn.datasets import make_classification as skl_make_classification

from biomedical_data_generator.utils.sklearn_compat import make_biomedical_dataset


# --------------------------------------------------------------------------- #
# Basic API compatibility: shapes, dtypes, output types
# --------------------------------------------------------------------------- #


def test_basic_numpy_shapes_and_dtypes():
    """make_biomedical_dataset should behave like a sklearn-style generator.

    - Returns NumPy arrays by default.
    - Shapes follow (n_samples, n_features) and (n_samples,).
    - X is floating, y is integer-like (classification labels).
    """
    n_samples = 120
    n_features = 10
    n_informative = 4

    X, y = make_biomedical_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        random_state=0,
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)

    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.integer)


def test_pandas_output_and_meta_roundtrip():
    """return_pandas / return_meta should produce DataFrame, Series and meta."""
    X, y, meta = make_biomedical_dataset(
        n_samples=50,
        n_features=8,
        n_informative=3,
        n_redundant=0,
        random_state=1,
        return_pandas=True,
        return_meta=True,
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    # Meta is a "black box" here, but it must be present and not None.
    assert meta is not None

    assert X.shape == (50, 8)
    assert y.shape == (50,)


# --------------------------------------------------------------------------- #
# Class weighting semantics: similar to sklearn.make_classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n_samples, weights",
    [
        (101, (0.5, 0.5)),
        (103, (0.7, 0.3)),
        (200, (0.2, 0.3, 0.5)),
    ],
)
def test_class_weights_semantics_close_to_expected(n_samples, weights):
    """Class weights should be translated into class sizes as documented.

    We do not require exact equality with sklearn's implementation,
    but we require:

    - all samples are assigned a class,
    - observed frequencies are close to the requested weights
      (off by at most ~1 sample due to rounding).
    """
    n_classes = len(weights)

    X, y = make_biomedical_dataset(
        n_samples=n_samples,
        n_features=10,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        weights=weights,
        random_state=0,
    )

    assert X.shape[0] == n_samples
    assert y.shape[0] == n_samples

    counts = np.bincount(y, minlength=n_classes)
    assert counts.sum() == n_samples

    expected = np.array(weights) / np.sum(weights) * n_samples
    # Allow at most ~1–2 samples deviation per class due to rounding.
    assert np.all(np.abs(counts - expected) <= 1.5)


def test_compared_to_sklearn_class_weights_rounding():
    """Class-size rounding should be compatible with sklearn in spirit.

    This is a weak compatibility test: both generators should produce
    similar class sizes for the same weights and n_samples, not wildly
    different ones.
    """
    n_samples = 137
    weights = (0.7, 0.3)
    n_classes = len(weights)

    X_skl, y_skl = skl_make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=list(weights),
        flip_y=0.0,
        random_state=42,
    )

    X_bdg, y_bdg = make_biomedical_dataset(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=n_classes,
        weights=weights,
        random_state=42,
    )

    counts_skl = np.bincount(y_skl, minlength=n_classes)
    counts_bdg = np.bincount(y_bdg, minlength=n_classes)

    # Both should sum to n_samples.
    assert counts_skl.sum() == n_samples
    assert counts_bdg.sum() == n_samples

    # Per-class difference should be small (same rounding strategy).
    assert np.all(np.abs(counts_skl - counts_bdg) <= 1)


# --------------------------------------------------------------------------- #
# Feature accounting: n_informative + n_redundant + n_noise == n_features
# --------------------------------------------------------------------------- #


def _count_redundant_proxies_from_meta(meta) -> int:
    """Infer the number of 'redundant' (proxy) features from DatasetMeta.

    This helper is written against the actual DatasetMeta API in this
    project:

    - `corr_cluster_indices`: dict[cid, list[int]]
    - `anchor_idx`: dict[cid, int | None]

    If these attributes are not present, we fall back to 0 to keep the
    tests robust against future changes in the meta structure.
    """
    # Prefer the BDG-specific name; fall back to any generic name if present.
    cluster_indices_attr = None
    if hasattr(meta, "corr_cluster_indices"):
        cluster_indices_attr = "corr_cluster_indices"
    elif hasattr(meta, "cluster_indices"):
        cluster_indices_attr = "cluster_indices"

    if cluster_indices_attr is None or not hasattr(meta, "anchor_idx"):
        return 0

    cluster_indices = getattr(meta, cluster_indices_attr)
    proxies = 0
    for cid, cols in cluster_indices.items():
        anchor = meta.anchor_idx.get(cid)
        if anchor is None:
            # Cluster without distinguished anchor → treat all as proxies.
            proxies += len(cols)
        else:
            # All non-anchor members are considered proxies.
            proxies += max(0, len(cols) - 1)
    return proxies


def test_feature_accounting_without_redundant():
    """When n_redundant=0, we expect only informative + noise features.

    We only require that:

    - total features match n_features,
    - informative + noise == n_features,
    - meta.informative_idx and meta.noise_idx cover all columns
      (no hidden extras), if those attributes exist.
    """
    n_features = 20
    n_informative = 5

    X, y, meta = make_biomedical_dataset(
        n_samples=80,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_noise=0,  # let wrapper infer remaining noise features
        random_state=0,
        return_meta=True,
    )

    assert X.shape == (80, n_features)

    # If the meta object exposes these attributes, we validate them.
    if hasattr(meta, "informative_idx") and hasattr(meta, "noise_idx"):
        n_inf_obs = len(meta.informative_idx)
        n_noise_obs = len(meta.noise_idx)

        assert n_inf_obs >= n_informative
        assert n_inf_obs + n_noise_obs == n_features


def test_feature_accounting_with_redundant_cluster():
    """n_redundant should map to proxies from a correlated cluster.

    The sklearn-style wrapper is expected to:

    - create one informative anchor cluster when n_redundant > 0,
    - contribute exactly n_redundant proxy features via this cluster.

    We verify this via DatasetMeta, if available.
    """
    n_features = 15
    n_informative = 4
    n_redundant = 3

    X, y, meta = make_biomedical_dataset(
        n_samples=60,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_noise=0,  # will be inferred as the remainder
        random_state=123,
        return_meta=True,
    )

    assert X.shape == (60, n_features)

    if hasattr(meta, "informative_idx") and hasattr(meta, "noise_idx"):
        n_inf_obs = len(meta.informative_idx)
        n_noise_obs = len(meta.noise_idx)
        n_proxies = _count_redundant_proxies_from_meta(meta)

        # Total number of features must decompose correctly.
        assert n_inf_obs + n_noise_obs + n_proxies == n_features
        # The number of proxies should match n_redundant by design.
        assert n_proxies == n_redundant


# --------------------------------------------------------------------------- #
# sklearn pipeline compatibility: can be used as drop-in dataset generator
# --------------------------------------------------------------------------- #


def test_works_in_sklearn_pipeline():
    """Generated data should plug into a standard sklearn pipeline.

    This is a practical compatibility test: the function should be usable
    wherever sklearn's make_classification is used to create toy data.
    """
    X, y = make_biomedical_dataset(
        n_samples=200,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        n_noise=0,
        random_state=0,
    )

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )

    # Fit without raising errors
    pipe.fit(X, y)

    # Quick sanity check: model should perform better than random guessing.
    y_pred = pipe.predict(X)
    ba = balanced_accuracy_score(y, y_pred)

    # Random guessing in binary classification → BA ~ 0.5.
    assert ba > 0.6
