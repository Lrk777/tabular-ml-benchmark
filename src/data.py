"""Load OpenML Adult and produce seeded stratified train/val/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class SplitData:
    """Hold feature/label matrices for train, val, and test."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    categorical: list[str]
    numeric: list[str]


def load_adult(data_id: int = 1590, target_column: str = "class") -> pd.DataFrame:
    """Fetch OpenML Adult and return a single DataFrame with the target column."""
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    frame = bunch.frame.copy()
    if target_column not in frame.columns:
        # Some OpenML exports name the target `class` already via bunch.target
        if bunch.target is not None and target_column not in frame.columns:
            frame[target_column] = bunch.target
        else:
            raise KeyError(f"Target column {target_column!r} not in {list(frame.columns)}")
    # Normalize string labels / strip whitespace
    frame[target_column] = frame[target_column].astype(str).str.strip()
    return frame


def encode_labels(y: pd.Series, positive_label: str = ">50K") -> np.ndarray:
    """Map income labels to {0, 1}; positive = >50K."""
    positive = positive_label.strip()
    # Handle both ">50K" and ">50K." variants from UCI/OpenML
    normalized = y.astype(str).str.strip().str.rstrip(".")
    pos_norm = positive.rstrip(".")
    return (normalized == pos_norm).astype(np.int64).to_numpy()


def make_splits(
    frame: pd.DataFrame,
    target_column: str = "class",
    positive_label: str = ">50K",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> SplitData:
    """Stratified train/val/test split. val_size is fraction of the *train* pool."""
    y = encode_labels(frame[target_column], positive_label=positive_label)
    X = frame.drop(columns=[target_column])

    categorical = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
    numeric = [c for c in X.columns if c not in categorical]

    # Coerce categoricals to string for stable one-hot; fill NA
    X = X.copy()
    for c in categorical:
        X[c] = X[c].astype("string").fillna("Missing").astype(str)
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    # val_size relative to remaining trainval
    relative_val = val_size
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        random_state=seed,
        stratify=y_trainval,
    )
    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        categorical=categorical,
        numeric=numeric,
    )


def build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """One-hot categoricals + scale numerics. Fits only on train later."""
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", ohe, categorical),
        ],
        remainder="drop",
    )


def load_and_split(config: dict[str, Any]) -> SplitData:
    """Convenience: load Adult from config and split."""
    frame = load_adult(
        data_id=int(config.get("openml_data_id", 1590)),
        target_column=config.get("target_column", "class"),
    )
    return make_splits(
        frame,
        target_column=config.get("target_column", "class"),
        positive_label=config.get("positive_label", ">50K"),
        test_size=float(config.get("test_size", 0.2)),
        val_size=float(config.get("val_size", 0.2)),
        seed=int(config.get("seed", 42)),
    )
