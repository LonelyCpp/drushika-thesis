"""
diagnostic_analysis.py

Computes per-class diagnostic metrics (sensitivity, specificity, PPV, NPV,
ROC-AUC) for the saved tuned LightGBM and baseline LightGBM models on the
held-out test split. Reports 95% CIs via stratified bootstrap, and compares
the two models' AUCs per class using DeLong's test.
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA_PATH = "./data.csv"
TARGET_COLUMN = "Risk"
RANDOM_STATE = 42
TEST_SIZE = 0.30
OUTPUT_DIR = "./model_outputs"
N_BOOT = 2000
ALPHA = 0.05

MODEL_FILES = {
    "Tuned LightGBM": "lgb_tuned_model.joblib",
    "Baseline LightGBM": "baseline_model.joblib",
}


def load_split():
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].astype(str).str.strip().replace({"N": "NO"})
    preproc = joblib.load(os.path.join(OUTPUT_DIR, "preprocessor.joblib"))
    Xt = preproc.transform(X)
    _, X_test, _, y_test = train_test_split(
        Xt, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return np.asarray(X_test), y_test.reset_index(drop=True)


def diag_from_cm(y_true_bin, y_pred_bin):
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return sens, spec, ppv, npv


def bootstrap_ci(y_true_bin, y_score, y_pred_bin, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true_bin)
    aucs, sens_l, spec_l, ppv_l, npv_l = [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true_bin[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        ys = y_score[idx]
        yp = y_pred_bin[idx]
        try:
            aucs.append(roc_auc_score(yt, ys))
        except ValueError:
            pass
        s, sp, pv, nv = diag_from_cm(yt, yp)
        sens_l.append(s)
        spec_l.append(sp)
        ppv_l.append(pv)
        npv_l.append(nv)

    def pct(arr):
        arr = np.array([a for a in arr if not np.isnan(a)])
        if len(arr) == 0:
            return (np.nan, np.nan)
        return (np.percentile(arr, 100 * ALPHA / 2), np.percentile(arr, 100 * (1 - ALPHA / 2)))

    return {
        "auc": pct(aucs),
        "sens": pct(sens_l),
        "spec": pct(spec_l),
        "ppv": pct(ppv_l),
        "npv": pct(npv_l),
    }


# ---------- DeLong's test (Sun & Xu 2014 fast implementation) ----------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_pvalue(y_true, score_a, score_b):
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    preds_sorted = np.vstack((score_a, score_b))[:, order]
    aucs, cov = _fast_delong(preds_sorted, label_1_count)
    if np.ndim(cov) == 0:
        cov = np.array([[cov, cov], [cov, cov]])
    L = np.array([[1, -1]])
    var = float(L @ cov @ L.T)
    if var <= 0:
        return aucs, 1.0
    z = float((L @ aucs)) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs, p


def main():
    X_test, y_test = load_split()
    classes = sorted(y_test.unique())
    print(f"Classes: {classes}, test n={len(y_test)}")

    scores = {}
    preds = {}
    for label, fname in MODEL_FILES.items():
        m = joblib.load(os.path.join(OUTPUT_DIR, fname))
        scores[label] = m.predict_proba(X_test)
        preds[label] = m.predict(X_test)
        # align column order to `classes`
        col_idx = [list(m.classes_).index(c) for c in classes]
        scores[label] = scores[label][:, col_idx]

    rows = []
    delong_rows = []
    for ci, c in enumerate(classes):
        y_bin = (y_test.values == c).astype(int)
        for label in MODEL_FILES:
            ys = scores[label][:, ci]
            yp = (preds[label] == c).astype(int)
            sens, spec, ppv, npv = diag_from_cm(y_bin, yp)
            try:
                auc = roc_auc_score(y_bin, ys)
            except ValueError:
                auc = np.nan
            ci_dict = bootstrap_ci(y_bin, ys, yp)
            rows.append({
                "class": c, "model": label,
                "AUC": auc, "AUC_CI": ci_dict["auc"],
                "Sens": sens, "Sens_CI": ci_dict["sens"],
                "Spec": spec, "Spec_CI": ci_dict["spec"],
                "PPV": ppv, "PPV_CI": ci_dict["ppv"],
                "NPV": npv, "NPV_CI": ci_dict["npv"],
            })

        # DeLong tuned vs baseline
        sa = scores["Tuned LightGBM"][:, ci]
        sb = scores["Baseline LightGBM"][:, ci]
        aucs_pair, p = delong_pvalue(y_bin, sa, sb)
        delong_rows.append({"class": c, "AUC_tuned": float(aucs_pair[0]),
                            "AUC_baseline": float(aucs_pair[1]), "p_value": p})

    print("\n=== Per-class metrics with 95% CIs ===")
    for r in rows:
        print(f"[{r['model']}] {r['class']:>8s}  "
              f"AUC={r['AUC']:.3f} ({r['AUC_CI'][0]:.3f}-{r['AUC_CI'][1]:.3f})  "
              f"Sens={r['Sens']:.3f} ({r['Sens_CI'][0]:.3f}-{r['Sens_CI'][1]:.3f})  "
              f"Spec={r['Spec']:.3f} ({r['Spec_CI'][0]:.3f}-{r['Spec_CI'][1]:.3f})  "
              f"PPV={r['PPV']:.3f} ({r['PPV_CI'][0]:.3f}-{r['PPV_CI'][1]:.3f})  "
              f"NPV={r['NPV']:.3f} ({r['NPV_CI'][0]:.3f}-{r['NPV_CI'][1]:.3f})")

    print("\n=== DeLong's test: Tuned vs Baseline (per class) ===")
    for r in delong_rows:
        sig = "*" if r["p_value"] < 0.05 else ""
        print(f"  {r['class']:>8s}  AUC tuned={r['AUC_tuned']:.4f}  "
              f"AUC base={r['AUC_baseline']:.4f}  p={r['p_value']:.4f} {sig}")

    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "diagnostic_metrics.csv"), index=False)
    pd.DataFrame(delong_rows).to_csv(os.path.join(OUTPUT_DIR, "delong_test.csv"), index=False)
    print(f"\nSaved diagnostic_metrics.csv and delong_test.csv to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
