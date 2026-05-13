# Case Study 1 -- AI4I 2020

Python implementation behind Case Study 1 of the GUIDE-PdM paper.
The script (`case1_ai4i.py`) is a direct realisation of the
case-specific pseudocode in Appendix B: every Phase 1-4 decision node
is either a fixed input or a numeric inequality, so the run is fully
deterministic.

## Files

| File               | Purpose                                          |
|--------------------|--------------------------------------------------|
| `case1_ai4i.py`    | end-to-end script, Phases 1-5                    |
| `ai4i2020.csv`     | AI4I 2020 dataset (10,000 rows)                  |
| `requirements.txt` | Python dependencies                              |
| `figures/`         | PNGs (created on first run)                      |
| `results/`         | `metrics.csv` and `decision_log.json`            |

## How to run

```
pip install -r requirements.txt
python case1_ai4i.py
```

Tested on Python 3.10 and 3.14. Run time is well under a minute.

## What the script does

- **Phase 1.** Acceptance gate: precision(+) >= 0.90, recall(+) >= 0.80.
  AUPRC is reported but not gated. The decision logic of the
  selected model must be inspectable.
- **Phase 2.** Data gate: max missing-value ratio <= 0.01; positive
  class share in [0.01, 0.10] triggers cost-sensitive training.
  Branch is set to 3B (data-driven).
- **Phase 3.** Three physics-informed features are added --
  `Power = (rpm * 2*pi/60) * Torque`, `dT = ProcessTemp - AirTemp`,
  `Strain = Torque * Tool wear`. `Type` is one-hot encoded. The
  candidates are Logistic Regression, Decision Tree (depths 4, 5, 6;
  smallest depth wins on ties), Random Forest, XGBoost, and an MLP.
- **Phase 3B.5.** The MLP is accepted only if it beats the best
  traditional model by >= 0.05 in F1 and >= 0.05 in recall. On AI4I
  this does not happen, and the decision log records the negative
  deltas.
- **Phase 4.** Selected model must clear the Phase-1 gate on the
  held-out test set and the depth limit for the tree.

Each model gets the same threshold-tuning protocol: pick the
probability threshold that maximises F1(+) on validation, freeze it,
apply unchanged to the test set.

## Outputs

`figures/` contains:

- `fig_class_imbalance.png`        -- class-imbalance bar
- `fig_feature_distributions.png`  -- per-feature histograms by class
- `fig_decision_tree.png`          -- the selected tree
- `AI4I_CM_DT.png`                 -- DT confusion matrix
- `AI4I_CM_MLP.png`                -- MLP confusion matrix
- `fig_confusion_matrices.png`     -- DT vs MLP side by side
- `fig_dt_feature_importance.png`  -- Gini importance for the DT
- `fig_pr_curves.png`              -- PR curves, DT vs MLP

`results/` contains:

- `metrics.csv`        -- one row per model with accuracy, precision,
  recall, F1, AUPRC, and a complexity tag.
- `decision_log.json`  -- thresholds, selected model, escalation
  verdict, and the gain deltas.

## Reproducibility

`SEED = 2026` seeds:
- numpy via `np.random.seed`,
- scikit-learn estimators via `random_state`,
- the oversampling RNG used to balance the MLP training set,
- xgboost via `random_state` and the deterministic
  `tree_method="hist"`.

BLAS threading on different machines can shift the last decimals of
the metrics but does not change the verdict.

Dataset source: Matzka, *Explainable Artificial Intelligence for
Predictive Maintenance Applications*, AI4I 2020.
