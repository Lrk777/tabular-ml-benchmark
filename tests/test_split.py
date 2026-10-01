"""Unit tests for split determinism and label encoding (no OpenML download)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import encode_labels, make_splits
from src.train import MajorityClassifier, _metrics


def _toy_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, size=n)
    hours = rng.integers(20, 60, size=n)
    workclass = rng.choice(["Private", "Gov", "Self"], size=n)
    # Imbalanced labels ~ 24% positive
    logits = 0.04 * (age - 40) + 0.03 * (hours - 40) + (workclass == "Self") * 0.5
    prob = 1 / (1 + np.exp(-logits))
    y = np.where(rng.random(n) < prob * 0.35 + 0.1, ">50K", "<=50K")
    return pd.DataFrame(
        {
            "age": age,
            "hours": hours,
            "workclass": workclass,
            "class": y,
        }
    )


def test_encode_labels_strips_dot_variant():
    y = pd.Series([">50K", "<=50K", ">50K.", " <=50K "])
    enc = encode_labels(y, positive_label=">50K")
    assert enc.tolist() == [1, 0, 1, 0]


def test_make_splits_deterministic():
    frame = _toy_frame()
    a = make_splits(frame, seed=42, test_size=0.2, val_size=0.2)
    b = make_splits(frame, seed=42, test_size=0.2, val_size=0.2)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_frame_equal(a.X_val, b.X_val)
    pd.testing.assert_frame_equal(a.X_test, b.X_test)
    np.testing.assert_array_equal(a.y_train, b.y_train)
    np.testing.assert_array_equal(a.y_val, b.y_val)
    np.testing.assert_array_equal(a.y_test, b.y_test)


def test_majority_classifier_metrics():
    y = np.array([0, 0, 0, 1])
    clf = MajorityClassifier().fit(np.zeros((4, 1)), y)
    pred = clf.predict(np.zeros((4, 1)))
    score = clf.predict_proba(np.zeros((4, 1)))[:, 1]
    assert set(pred.tolist()) == {0}
    m = _metrics(y, pred, score)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.0)
    assert m["roc_auc"] == pytest.approx(0.5)


@pytest.mark.slow
def test_openml_adult_load_smoke():
    from src.data import load_adult

    frame = load_adult(data_id=1590)
    assert "class" in frame.columns or frame.shape[1] > 5
    assert len(frame) > 10_000
