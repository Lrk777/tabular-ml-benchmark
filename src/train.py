"""Train baselines and stronger models; select by validation ROC-AUC."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data import build_preprocessor, load_and_split


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Predict the majority training class; scores are constant prior."""

    def fit(self, X, y):
        y = np.asarray(y)
        vals, counts = np.unique(y, return_counts=True)
        self.classes_ = vals
        self.majority_ = vals[np.argmax(counts)]
        self.prior_ = float(counts[np.argmax(counts)] / len(y))
        # P(y=1) for predict_proba column order
        self.pos_rate_ = float(np.mean(y == 1)) if 1 in vals else 0.0
        return self

    def predict(self, X):
        return np.full(shape=len(X), fill_value=self.majority_, dtype=int)

    def predict_proba(self, X):
        p1 = self.pos_rate_
        proba = np.column_stack([1.0 - p1, p1])
        return np.broadcast_to(proba, (len(X), 2)).copy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def build_models(cfg: dict[str, Any], seed: int) -> dict[str, BaseEstimator]:
    """Instantiate named estimators from config (without preprocessor)."""
    models_cfg = cfg.get("models", {})
    out: dict[str, BaseEstimator] = {}

    if "majority" in models_cfg:
        out["majority"] = MajorityClassifier()

    if "logistic" in models_cfg:
        p = models_cfg["logistic"] or {}
        out["logistic"] = LogisticRegression(
            max_iter=int(p.get("max_iter", 1000)),
            class_weight=p.get("class_weight", "balanced"),
            random_state=seed,
            solver="lbfgs",
        )

    if "random_forest" in models_cfg:
        p = models_cfg["random_forest"] or {}
        out["random_forest"] = RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=p.get("max_depth"),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            class_weight=p.get("class_weight", "balanced_subsample"),
            n_jobs=int(p.get("n_jobs", -1)),
            random_state=seed,
        )

    if "hist_gradient_boosting" in models_cfg:
        p = models_cfg["hist_gradient_boosting"] or {}
        out["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_iter=int(p.get("max_iter", 200)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            max_depth=p.get("max_depth"),
            l2_regularization=float(p.get("l2_regularization", 0.0)),
            random_state=seed,
        )

    if not out:
        raise ValueError("No models configured under configs/*.yaml `models:`")
    return out


def _predict_scores(pipe: Pipeline, X) -> tuple[np.ndarray, np.ndarray]:
    pred = pipe.predict(X)
    if hasattr(pipe, "predict_proba"):
        score = pipe.predict_proba(X)[:, 1]
    else:
        score = pred.astype(float)
    return pred, score


def train_all(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Fit all models, score val/test, pick best by val selection metric."""
    seed = int(config.get("seed", 42))
    np.random.seed(seed)

    split = load_and_split(config)
    models = build_models(config, seed=seed)
    selection_metric = config.get("selection_metric", "roc_auc")

    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    results: dict[str, Any] = {
        "seed": seed,
        "openml_data_id": config.get("openml_data_id", 1590),
        "selection_metric": selection_metric,
        "n_train": int(len(split.y_train)),
        "n_val": int(len(split.y_val)),
        "n_test": int(len(split.y_test)),
        "models": {},
    }

    best_name = None
    best_val = -np.inf
    pred_arrays: dict[str, np.ndarray] = {"y_test": split.y_test}

    for name, est in models.items():
        # Fresh preprocessor per pipeline so each model owns its fitted state
        pipe = Pipeline(
            [
                ("pre", build_preprocessor(split.categorical, split.numeric)),
                ("clf", est),
            ]
        )
        pipe.fit(split.X_train, split.y_train)

        yhat_val, score_val = _predict_scores(pipe, split.X_val)
        yhat_test, score_test = _predict_scores(pipe, split.X_test)
        val_m = _metrics(split.y_val, yhat_val, score_val)
        test_m = _metrics(split.y_test, yhat_test, score_test)

        results["models"][name] = {"val": val_m, "test": test_m}
        joblib.dump(pipe, models_dir / f"{name}.joblib")
        pred_arrays[f"{name}_pred"] = yhat_test
        pred_arrays[f"{name}_score"] = score_test

        metric_val = val_m.get(selection_metric, val_m["roc_auc"])
        if metric_val > best_val:
            best_val = metric_val
            best_name = name

    results["best_model"] = best_name
    results["best_val_metric"] = float(best_val)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(run_dir / "predictions.npz", **pred_arrays)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Adult income models")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output directory (default: reports/latest)",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reports_root = Path(config.get("reports_dir", "reports"))
    run_dir = args.run_dir or (reports_root / "latest")

    # Fresh latest run
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    # Copy config for provenance
    shutil.copy(args.config, run_dir / "config.yaml")

    results = train_all(config, run_dir)
    print(json.dumps({"best_model": results["best_model"], "path": str(run_dir)}, indent=2))
    for name, block in results["models"].items():
        t = block["test"]
        print(
            f"{name:24s}  acc={t['accuracy']:.3f}  f1={t['f1']:.3f}  "
            f"roc_auc={t['roc_auc']:.3f}  ap={t['average_precision']:.3f}"
        )


if __name__ == "__main__":
    main()
