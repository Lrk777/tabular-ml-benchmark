# tabular-ml-benchmark

Honest baselines and model comparison on the **UCI Adult** income dataset (OpenML `data_id=1590`).

## Problem

Can we predict whether an adult earns **>50K** vs **≤50K** from census-style tabular features, and beat a majority-class baseline with a fair train/val/test protocol?

## Demo

```text
$ python -m src.train --config configs/default.yaml
$ python -m src.evaluate --run-dir reports/latest

model                      acc      f1  roc_auc      ap
majority                 0.761   0.000    0.500   0.239
logistic                 0.809   0.678    0.905   0.765
random_forest            0.807   0.684    0.913   0.789
hist_gradient_boosting   0.877   0.718    0.929   0.831 *
```

Figures land in `reports/figures/` (ROC, PR, confusion matrix, AUC bars).

## Method

1. Load OpenML Adult **1590**; binary target `>50K` / `<=50K`.
2. Stratified **train / val / test** (seed `42`; test 20%, then val 20% of the remainder).
3. Preprocess: median-impute + scale numerics; one-hot categoricals (`Missing` for NA).
4. Fit four models on train only:
   - **majority** — always predict the majority class
   - **logistic** — `LogisticRegression` with `class_weight='balanced'`
   - **random_forest** — RF with balanced subsample weights
   - **hist_gradient_boosting** — `HistGradientBoostingClassifier`
5. Select the best model by **validation ROC-AUC**; report **test** metrics (accuracy, F1, ROC-AUC, average precision).

Config lives in `configs/default.yaml` so seeds, split sizes, and hyperparameters are explicit.

## Results

Checked-in test metrics (seed **42**):

| Model | Acc | F1 | ROC-AUC | AP |
|-------|-----|-----|---------|-----|
| majority | 0.761 | 0.000 | 0.500 | 0.239 |
| logistic | 0.809 | 0.678 | 0.905 | 0.765 |
| random_forest | 0.807 | 0.684 | 0.913 | 0.789 |
| **hist_gradient_boosting** | **0.877** | **0.718** | **0.929** | **0.831** |

Hist gradient boosting wins on every ranking metric while the majority baseline confirms the ~76% accuracy floor from class imbalance.

## Stack

- Python 3.11+
- pandas, NumPy, scikit-learn, matplotlib, seaborn, PyYAML, joblib, pytest

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m src.train --config configs/default.yaml
python -m src.evaluate --run-dir reports/latest
pytest -q -m "not slow"
```

First train downloads Adult via OpenML (cached under `~/scikit_learn_data`).

## Limitations

- **Class imbalance:** positives are ~24%; accuracy alone is misleading — hence ROC-AUC / AP / F1.
- **Leakage risk:** no target encoding; all transforms fit on train only. Still, Adult is a *lab* dataset — results are not a claim about modern income prediction.
- **No calibration / fairness audit** in this MVP (education, sex, race slices are left for follow-up).
- Metrics may drift slightly across sklearn minor versions; seed and config are fixed.

## License

MIT for this repository’s code. Adult / OpenML data retain their original terms — cite OpenML dataset 1590 / UCI Adult when publishing results.

## Cite

```text
OpenML Adult (data_id=1590); UCI Machine Learning Repository — Adult Data Set.
```
