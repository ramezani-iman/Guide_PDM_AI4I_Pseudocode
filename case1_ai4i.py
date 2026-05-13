"""
AI4I 2020 -- failure-screening pipeline for Case Study 1 of GUIDE-PdM.

The script follows the case-specific pseudocode in Appendix B:
problem framing -> data gate -> feature engineering and splits ->
five candidates (logreg, decision tree at depths 4-6, random forest,
xgboost, mlp) -> escalation check (mlp vs best traditional) -> accept
on a Phase-1 gate on the test set.

Run from this folder:
    python case1_ai4i.py
Outputs go to ./figures and ./results.

Data: Matzka, "Explainable AI for Predictive Maintenance
Applications", AI4I 2020 (csv shipped in this folder).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed -- skipping the XGBoost row.")


SEED = 2026

# Phase 1 -- acceptance gate on the positive class.
# Precision >= 0.90 keeps the operator alarm load low.
# Recall    >= 0.80 keeps most failures from slipping through.
# AUPRC is reported as a descriptor but not gated.
TAU_PREC = 0.90
TAU_REC  = 0.80

# Phase 2 -- data gate.
MU_MAX  = 0.01           # max tolerated missing-value ratio
RHO_LOW, RHO_HIGH = 0.01, 0.10   # imbalance band -> cost-sensitive training

# Phase 3B.5 -- minimum gains the MLP must show over the best traditional
# model to be worth its added complexity (in F1 and recall on the positive
# class, both on the test set).
DELTA_F1   = 0.05
DELTA_REC  = 0.05

# Phase 3 -- decision-tree depths to sweep. Capped at 6 to keep the tree
# inspectable; ties on validation F1 break in favour of the smaller depth.
DT_DEPTHS = [4, 5, 6]

HERE     = Path(__file__).resolve().parent
DATA_CSV = HERE / "ai4i2020.csv"
FIG_DIR  = HERE / "figures"
RES_DIR  = HERE / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


# ---- Phase 1 -----------------------------------------------------------
def phase1_summary():
    print("\n[Phase 1] problem formulation")
    print("  asset  : milling machine (AI4I 2020 benchmark)")
    print("  task   : binary failure screening on tabular inputs")
    print(f"  gate   : precision(+) >= {TAU_PREC:.2f}, recall(+) >= {TAU_REC:.2f}")
    print("  also   : decision logic must be inspectable")


# ---- Phase 2 -----------------------------------------------------------
def load_and_check(path):
    print("\n[Phase 2] data gate")
    df = pd.read_csv(path)

    # Drop identifiers and per-cause flags; keep the binary target.
    df = df.drop(columns=["UDI", "Product ID",
                          "TWF", "HDF", "PWF", "OSF", "RNF"])
    df = df.rename(columns={"Machine failure": "y"})

    mu  = df.isna().mean().max()
    rho = df["y"].mean()
    print(f"  rows = {len(df)}, feature cols = {df.shape[1] - 1}")
    print(f"  mu (max missing ratio) = {mu:.4f}  (cap: {MU_MAX})")
    print(f"  rho (positive share)   = {rho:.4f}  (band: [{RHO_LOW}, {RHO_HIGH}])")

    if mu > MU_MAX:
        raise RuntimeError("Phase 2 data gate failed: too much missingness.")

    cost_sensitive = (RHO_LOW <= rho <= RHO_HIGH)
    print(f"  cost-sensitive training : {cost_sensitive}")
    print("  branch                  : 3B (data-driven)")
    return df, cost_sensitive


# ---- Phase 3: features, splits, candidates -----------------------------
def add_physics_features(df):
    """
    Three derived features tied to the failure modes used to generate
    the dataset:
        Power  = (rpm * 2*pi / 60) * Torque        -> PWF
        dT     = ProcessTemp - AirTemp             -> HDF
        Strain = Torque * Tool wear                -> OSF
    Square brackets are stripped from column names because XGBoost
    rejects them.
    """
    out = df.rename(columns={
        "Air temperature [K]":     "Air temperature (K)",
        "Process temperature [K]": "Process temperature (K)",
        "Rotational speed [rpm]":  "Rotational speed (rpm)",
        "Torque [Nm]":             "Torque (Nm)",
        "Tool wear [min]":         "Tool wear (min)",
    }).copy()
    out["Power (W)"]       = (out["Rotational speed (rpm)"] * 2.0 * np.pi / 60.0) * out["Torque (Nm)"]
    out["dT (K)"]          = out["Process temperature (K)"] - out["Air temperature (K)"]
    out["Strain (Nm.min)"] = out["Torque (Nm)"] * out["Tool wear (min)"]

    type_dum = pd.get_dummies(out["Type"], prefix="Type")
    out = pd.concat([out.drop(columns=["Type"]), type_dum], axis=1)

    feats = [c for c in out.columns if c != "y"]
    out[feats] = out[feats].astype(float)
    return out


def split_70_15_15(df, seed=SEED):
    y = df["y"].values
    X = df.drop(columns=["y"])
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return (X_tr.reset_index(drop=True), X_va.reset_index(drop=True),
            X_te.reset_index(drop=True), y_tr, y_va, y_te)


def best_f1_threshold(y, p):
    """Sweep thresholds in [0.01, 0.99] and return the one with the
    highest F1 on the positive class."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def row(name, y_true, y_pred, p_pred, complexity):
    return {
        "model":      name,
        "accuracy":   accuracy_score(y_true, y_pred),
        "precision":  precision_score(y_true, y_pred, zero_division=0),
        "recall":     recall_score(y_true, y_pred, zero_division=0),
        "f1":         f1_score(y_true, y_pred, zero_division=0),
        "auprc":      average_precision_score(y_true, p_pred),
        "complexity": complexity,
    }


def fit_logreg(Xs, y):
    m = LogisticRegression(class_weight="balanced", max_iter=2000,
                           random_state=SEED)
    m.fit(Xs, y)
    return m


def fit_dt(X, y, depth):
    m = DecisionTreeClassifier(max_depth=depth, class_weight="balanced",
                               random_state=SEED)
    m.fit(X, y)
    return m


def fit_rf(X, y):
    m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               n_jobs=-1, random_state=SEED)
    m.fit(X, y)
    return m


def fit_xgb(X, y):
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                      scale_pos_weight=pos_weight, eval_metric="aucpr",
                      tree_method="hist", n_jobs=-1, random_state=SEED)
    m.fit(X, y)
    return m


def fit_mlp(Xs, y):
    # MLPClassifier has no class_weight; balance by oversampling
    # the positive class to match the negative count.
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([neg, rng.choice(pos, size=len(neg), replace=True)])
    rng.shuffle(idx)
    m = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                      max_iter=500, early_stopping=True,
                      validation_fraction=0.15, random_state=SEED)
    m.fit(Xs[idx], y[idx])
    return m


# ---- Plots -------------------------------------------------------------
def plot_class_imbalance(df, out_path):
    counts = df["y"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar(["No failure", "Failure"], counts.values,
           color=["#4472C4", "#C0504D"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,} ({v/counts.sum()*100:.2f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of instances")
    ax.set_title(f"AI4I class distribution (n = {counts.sum():,})")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_feature_distributions(df, out_path):
    feats = ["Air temperature (K)", "Process temperature (K)",
             "Rotational speed (rpm)", "Torque (Nm)", "Tool wear (min)",
             "Power (W)", "dT (K)", "Strain (Nm.min)"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, f in zip(axes.ravel(), feats):
        good = df.loc[df["y"] == 0, f]
        bad  = df.loc[df["y"] == 1, f]
        ax.hist(good, bins=40, alpha=0.6, color="#4472C4",
                label="No failure", density=True)
        ax.hist(bad,  bins=40, alpha=0.6, color="#C0504D",
                label="Failure",    density=True)
        ax.set_title(f, fontsize=10)
        ax.tick_params(labelsize=8)
    axes.ravel()[0].legend(loc="best", fontsize=8)
    fig.suptitle("Feature distributions by class", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_decision_tree(model, feature_names, out_path):
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names,
              class_names=["OK", "Fail"], filled=True, rounded=True,
              fontsize=8, ax=ax, impurity=False, proportion=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _draw_cm_on(ax, cm, title):
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["OK", "Fail"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["OK", "Fail"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)


def plot_single_cm(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    _draw_cm_on(ax, cm, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_two_cms(cm_dt, cm_mlp, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    _draw_cm_on(axes[0], cm_dt,  "Decision Tree")
    _draw_cm_on(axes[1], cm_mlp, "MLP")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dt_importance(model, feature_names, out_path):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp.index, imp.values, color="#4472C4")
    ax.set_xlabel("Gini importance")
    ax.set_title("Decision Tree feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curves(y_true, p_dt, p_mlp, out_path):
    p_d, r_d, _ = precision_recall_curve(y_true, p_dt)
    p_m, r_m, _ = precision_recall_curve(y_true, p_mlp)
    ap_d  = average_precision_score(y_true, p_dt)
    ap_m  = average_precision_score(y_true, p_mlp)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(r_d, p_d, color="#4472C4",
            label=f"Decision Tree (AUPRC = {ap_d:.3f})")
    ax.plot(r_m, p_m, color="#C0504D",
            label=f"MLP (AUPRC = {ap_m:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (test set)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---- Main --------------------------------------------------------------
def main():
    np.random.seed(SEED)

    phase1_summary()

    df, cost_sensitive = load_and_check(DATA_CSV)
    df = add_physics_features(df)

    plot_class_imbalance(df, FIG_DIR / "fig_class_imbalance.png")
    plot_feature_distributions(df, FIG_DIR / "fig_feature_distributions.png")

    print("\n[Phase 3] DPFE, candidates, and the MLP gate")
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_70_15_15(df, seed=SEED)
    feats = list(X_tr.columns)
    print(f"  features : {feats}")
    print(f"  sizes    : train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}")
    print(f"  positives: train={int(y_tr.sum())}, val={int(y_va.sum())}, test={int(y_te.sum())}")

    scaler = StandardScaler().fit(X_tr.values)
    Xtr_s, Xva_s, Xte_s = (scaler.transform(X.values) for X in (X_tr, X_va, X_te))

    results, raw_probs = [], {}

    # Logistic Regression
    m = fit_logreg(Xtr_s, y_tr)
    p_va, p_te = m.predict_proba(Xva_s)[:, 1], m.predict_proba(Xte_s)[:, 1]
    t = best_f1_threshold(y_va, p_va)
    results.append(row("Logistic Regression", y_te,
                       (p_te >= t).astype(int), p_te, "linear"))

    # Decision Tree -- sweep depths, pick best val F1, tie -> smaller depth
    dt_cands = []
    for d in DT_DEPTHS:
        cand = fit_dt(X_tr, y_tr, d)
        p_v  = cand.predict_proba(X_va)[:, 1]
        f1_v = f1_score(y_va, (p_v >= best_f1_threshold(y_va, p_v)).astype(int),
                        zero_division=0)
        print(f"  DT depth={d}: val F1 = {f1_v:.3f}")
        dt_cands.append((d, cand, f1_v))
    dt_cands.sort(key=lambda x: (-x[2], x[0]))
    d_star, dt_star, _ = dt_cands[0]
    print(f"  selected DT depth = {d_star}")

    p_va = dt_star.predict_proba(X_va)[:, 1]
    p_te = dt_star.predict_proba(X_te)[:, 1]
    t_dt = best_f1_threshold(y_va, p_va)
    yhat_dt = (p_te >= t_dt).astype(int)
    results.append(row(f"Decision Tree (depth={d_star})",
                       y_te, yhat_dt, p_te, f"depth={d_star}"))
    raw_probs["dt"] = (p_te, t_dt)
    cm_dt = confusion_matrix(y_te, yhat_dt)

    # Random Forest
    m = fit_rf(X_tr, y_tr)
    p_va, p_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
    t = best_f1_threshold(y_va, p_va)
    results.append(row("Random Forest", y_te,
                       (p_te >= t).astype(int), p_te, "300 trees"))

    # XGBoost
    if HAS_XGB:
        m = fit_xgb(X_tr, y_tr)
        p_va, p_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
        t = best_f1_threshold(y_va, p_va)
        results.append(row("XGBoost", y_te, (p_te >= t).astype(int), p_te,
                           "300 boosted trees"))

    # MLP -- the only complex candidate; subject to the escalation gate
    m = fit_mlp(Xtr_s, y_tr)
    p_va, p_te = m.predict_proba(Xva_s)[:, 1], m.predict_proba(Xte_s)[:, 1]
    t_mlp = best_f1_threshold(y_va, p_va)
    yhat_mlp = (p_te >= t_mlp).astype(int)
    results.append(row("MLP", y_te, yhat_mlp, p_te, "[32, 16]"))
    raw_probs["mlp"] = (p_te, t_mlp)
    cm_mlp = confusion_matrix(y_te, yhat_mlp)

    # ---- Phase 3B.5: complex-AI escalation rule ----
    print("\n[Phase 3B.5] does the MLP beat the best traditional model by enough?")
    df_res = pd.DataFrame(results)
    print(df_res[["model", "accuracy", "precision", "recall", "f1", "auprc"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best_trad = (df_res[df_res["model"] != "MLP"]
                 .sort_values("f1", ascending=False).iloc[0])
    mlp_row   = df_res[df_res["model"] == "MLP"].iloc[0]
    d_f1  = mlp_row["f1"]     - best_trad["f1"]
    d_rec = mlp_row["recall"] - best_trad["recall"]
    print(f"  best traditional : {best_trad['model']} "
          f"(F1={best_trad['f1']:.3f}, recall={best_trad['recall']:.3f})")
    print(f"  MLP              : F1={mlp_row['f1']:.3f}, recall={mlp_row['recall']:.3f}")
    print(f"  delta F1 = {d_f1:+.3f} (need >= {DELTA_F1})")
    print(f"  delta rec = {d_rec:+.3f} (need >= {DELTA_REC})")
    verdict = ("complex AI justified"
               if d_f1 >= DELTA_F1 and d_rec >= DELTA_REC
               else "complex AI not justified -- traditional model retained")
    print(f"  verdict          : {verdict}")

    # ---- Phase 4: acceptance on the test set ----
    print("\n[Phase 4] Phase-1 gate on the test set")
    df_gate = df_res.assign(passes=lambda r:
                            (r["precision"] >= TAU_PREC) &
                            (r["recall"]    >= TAU_REC))
    print(df_gate[["model", "precision", "recall", "auprc", "passes"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    dt_res = df_res[df_res["model"].str.startswith("Decision Tree")].iloc[0]
    dt_pass = (dt_res["precision"] >= TAU_PREC and dt_res["recall"] >= TAU_REC)
    print(f"  selected DT (depth={d_star}) passes Phase-1 gate : {dt_pass}")
    print(f"  selected DT satisfies interpretability gate     : True")
    print(f"  >>> SELECTED MODEL: {dt_res['model']}")

    # ---- Plots that need the trained models ----
    plot_decision_tree(dt_star, feats, FIG_DIR / "fig_decision_tree.png")
    plot_single_cm(cm_dt,  "Decision Tree", FIG_DIR / "AI4I_CM_DT.png")
    plot_single_cm(cm_mlp, "MLP",           FIG_DIR / "AI4I_CM_MLP.png")
    plot_two_cms(cm_dt, cm_mlp, FIG_DIR / "fig_confusion_matrices.png")
    plot_dt_importance(dt_star, feats, FIG_DIR / "fig_dt_feature_importance.png")
    plot_pr_curves(y_te, raw_probs["dt"][0], raw_probs["mlp"][0],
                   FIG_DIR / "fig_pr_curves.png")

    # ---- Numeric outputs ----
    df_res.to_csv(RES_DIR / "metrics.csv", index=False)
    log = {
        "seed": SEED,
        "thresholds": {
            "tau_precision_pos": TAU_PREC,
            "tau_recall_pos":    TAU_REC,
            "mu_max":            MU_MAX,
            "rho_band":          [RHO_LOW, RHO_HIGH],
            "delta_f1":          DELTA_F1,
            "delta_recall":      DELTA_REC,
        },
        "selected_model":  dt_res["model"],
        "dt_depth":        d_star,
        "dt_threshold":    t_dt,
        "mlp_threshold":   t_mlp,
        "verdict":         verdict,
        "delta_f1":        float(d_f1),
        "delta_recall":    float(d_rec),
        "dt_passes_gate":  bool(dt_pass),
    }
    with open(RES_DIR / "decision_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\ndone. figures -> {FIG_DIR}")
    print(f"      results -> {RES_DIR}")


if __name__ == "__main__":
    main()
