"""
evaluate_models.py

Loads the saved preprocessing pipeline and trained models from model_outputs,
recreates the same train/test split used in trainer.py, reports accuracy metrics,
and generates thesis-ready visualizations (confusion matrix image, feature
importance CSV + plot, and SHAP summary plot) for the strongest available model.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

try:
    import shap

    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = None
except ImportError as exc:
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = str(exc)

# Keep these settings aligned with trainer.py to ensure comparable splits
DATA_PATH = "./data.csv"
TARGET_COLUMN = "Risk"
RANDOM_STATE = 42
TEST_SIZE = 0.30
OUTPUT_DIR = "./model_outputs"
PREPROCESSOR_FILE = "preprocessor.joblib"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Baseline LightGBM": "baseline_model.joblib",
    "Tuned LightGBM": "lgb_tuned_model.joblib",
    "Baseline XGBoost": "xgb_baseline_model.joblib",
    "Tuned XGBoost": "xgb_tuned_model.joblib",
}
XGB_LABEL_ENCODER_FILE = "xgb_label_encoder.joblib"
MODEL_PRIORITY = [
    "Tuned LightGBM",
    "Tuned XGBoost",
    "Baseline LightGBM",
    "Baseline XGBoost",
    "Logistic Regression",
]
# Models included in the paper-style multi-model comparison (one row per algorithm)
PAPER_STYLE_MODELS = ["Logistic Regression", "Tuned LightGBM", "Tuned XGBoost"]
PAPER_STYLE_DISPLAY = {
    "Logistic Regression": "Logistic Regression",
    "Tuned LightGBM": "LightGBM",
    "Tuned XGBoost": "XGBoost",
}

CONFUSION_MATRIX_IMG = "confusion_matrix.png"
FEATURE_IMPORTANCE_CSV = "feature_importances.csv"
FEATURE_IMPORTANCE_IMG = "feature_importance.png"
SHAP_SUMMARY_IMG = "shap_summary.png"
ROC_SUMMARY_CSV = "roc_auc_summary.csv"
ROC_COMPARISON_IMG = "roc_comparison.png"
PAPER_METRICS_CSV = "paper_style_metrics.csv"
PAPER_METRICS_BAR_IMG = "model_performance_bars.png"
PAPER_ROC_PANEL_IMG = "roc_curves_all_models.png"

# Filename slug per model label (used for per-model ROC plots)
ROC_PLOT_SLUGS = {
    "Baseline LightGBM": "lgb_baseline",
    "Tuned LightGBM": "lgb_tuned",
    "Baseline XGBoost": "xgb_baseline",
    "Tuned XGBoost": "xgb_tuned",
}


def load_dataset():
    """Load the dataset and clean the target column exactly like trainer.py."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA_PATH '{DATA_PATH}' does not exist.")

    read_kwargs = {}
    if DATA_PATH.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(DATA_PATH, sheet_name=None, **read_kwargs)
        if isinstance(df, dict):
            df = next(iter(df.values()))
    else:
        df = pd.read_csv(DATA_PATH, **read_kwargs)

    df.columns = [str(c).strip() for c in df.columns]
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"TARGET_COLUMN '{TARGET_COLUMN}' not in dataset columns {df.columns.tolist()}")

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    if y.dtype == object:
        y = y.astype(str).str.strip()
        y = y.replace({"N": "NO", "": np.nan})
    if pd.api.types.is_float_dtype(y) and y.dropna().apply(float.is_integer).all():
        y = y.astype("Int64")

    if y.isna().any():
        missing_idx = y[y.isna()].index
        X = X.drop(index=missing_idx)
        y = y.drop(index=missing_idx)

    return X.reset_index(drop=True), y.reset_index(drop=True)


def prepare_test_split(preproc, X, y):
    """Apply saved preprocessing and recreate the deterministic train/test split."""
    X_transformed = preproc.transform(X)
    value_counts = y.value_counts()
    if y.nunique() <= 10 and value_counts.min() >= 2:
        stratify_arg = y
    else:
        stratify_arg = None

    _, X_test, _, y_test = train_test_split(
        X_transformed,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_arg,
    )
    return np.asarray(X_test), y_test


def extract_feature_names(preproc):
    """Recreate feature names order from the fitted ColumnTransformer."""
    feature_names = []
    for name, _, cols in getattr(preproc, "transformers_", []):
        if name == "remainder" or cols is None:
            continue
        if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            feature_names.extend([str(c) for c in cols])
        else:
            feature_names.append(str(cols))
    if not feature_names and hasattr(preproc, "feature_names_in_"):
        feature_names = [str(c) for c in preproc.feature_names_in_]
    if feature_names and hasattr(preproc, "transformers_") and feature_names[-1] == "remainder":
        feature_names = feature_names[:-1]
    return feature_names


def evaluate_model(model, X_test, y_test, label, label_encoder=None):
    """Print metrics for a single model and return predictions."""
    y_pred = model.predict(X_test)
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)
    print(f"\n===== {label} =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return y_pred


def save_confusion_matrix_plot(y_true, y_pred, output_path, title):
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap=plt.cm.Blues,
        colorbar=False,
    )
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix image saved to {output_path}")


def save_feature_importance_outputs(model, feature_names, csv_path, fig_path):
    if not hasattr(model, "feature_importances_"):
        print("Skipping feature importance plot/CSV (model lacks feature_importances_).")
        return

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        print("Feature importance dimension mismatch; skipping plot/CSV.")
        return

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fi_df.to_csv(csv_path, index=False)
    print(f"Feature importance CSV saved to {csv_path}")

    top_n = min(20, len(fi_df))
    top_df = fi_df.head(top_n).iloc[::-1]  # reverse for horizontal plot
    plt.figure(figsize=(8, max(4, top_n * 0.4)))
    plt.barh(top_df["feature"], top_df["importance"], color="#1f77b4")
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to {fig_path}")


def save_shap_summary_plot(model, X_test, feature_names, output_path):
    if not SHAP_AVAILABLE:
        print(f"Skipping SHAP summary plot (shap not available: {SHAP_IMPORT_ERROR}).")
        return

    if not hasattr(model, "booster_") and "LGBM" not in model.__class__.__name__:
        print("Skipping SHAP summary plot (only supported for LightGBM models).")
        return

    try:
        sample_size = min(500, X_test.shape[0])
        if sample_size == 0:
            print("Skipping SHAP summary plot (empty test set).")
            return

        # Convert to numpy array and ensure it's 2D
        if isinstance(X_test, pd.DataFrame):
            X_sample = X_test.iloc[:sample_size].values
        else:
            X_sample = np.asarray(X_test[:sample_size])
        
        if X_sample.ndim != 2:
            print("Skipping SHAP summary plot (unexpected feature array shape).")
            return

        # Get the actual number of features the model expects
        if hasattr(model, 'n_features_in_'):
            n_model_features = model.n_features_in_
        elif hasattr(model, 'n_features_'):
            n_model_features = model.n_features_
        else:
            n_model_features = X_sample.shape[1]

        print(f"Debug: X_sample shape: {X_sample.shape}, model expects: {n_model_features} features")

        # Create explainer with the model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        print(f"Debug: Raw SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"Debug: SHAP values is a list with {len(shap_values)} elements")
            print(f"Debug: First element shape: {shap_values[0].shape}")
        else:
            print(f"Debug: SHAP values shape: {shap_values.shape}")
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # For multi-class, use the first class as representative.
            shap_values_to_plot = shap_values[0]
            class_label = " (HIGH class)"
        elif shap_values.ndim == 3:
            # Shape is (samples, features, classes) - take first class
            print(f"Debug: SHAP values is 3D with shape {shap_values.shape}, extracting first class")
            shap_values_to_plot = shap_values[:, :, 0]
            class_label = " (HIGH class)"
        else:
            shap_values_to_plot = shap_values
            class_label = ""

        print(f"Debug: Final SHAP values shape after extraction: {shap_values_to_plot.shape}")

        n_shap_features = shap_values_to_plot.shape[1]
        n_data_features = X_sample.shape[1]
        
        # Prefer the real feature names from the preprocessor over generic
        # placeholders (LightGBM stores Column_0/1/... when fit on a NumPy array).
        def _looks_generic(names):
            if not names:
                return True
            import re
            return all(re.fullmatch(r"(Column|feature|f)_?\d+", str(n)) for n in names)

        model_names = None
        if hasattr(model, 'feature_name_') and model.feature_name_:
            model_names = list(model.feature_name_)
        elif hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            model_names = list(model.feature_names_in_)

        if feature_names and not _looks_generic(feature_names):
            feature_names_working = list(feature_names)
        elif model_names and not _looks_generic(model_names):
            feature_names_working = model_names
        elif model_names:
            feature_names_working = model_names
        elif feature_names:
            feature_names_working = list(feature_names)
        else:
            feature_names_working = [f"feature_{i}" for i in range(n_shap_features)]
        
        n_name_features = len(feature_names_working)
        print(f"Debug: Feature names count: {n_name_features}, names: {feature_names_working}")

        # Ensure all dimensions match
        n_common = min(n_shap_features, n_data_features, n_name_features)
        if n_common == 0:
            print("Skipping SHAP summary plot (no overlapping features).")
            return
        
        if n_common < max(n_shap_features, n_data_features, n_name_features):
            print(
                f"Warning: Feature dimension mismatch for SHAP "
                f"(shap={n_shap_features}, data={n_data_features}, names={n_name_features}). "
                f"Using first {n_common} features."
            )

        # Trim all arrays to the common dimension
        shap_values_trimmed = shap_values_to_plot[:, :n_common]
        X_sample_trimmed = X_sample[:, :n_common]
        feature_names_trimmed = feature_names_working[:n_common]

        print(f"Debug: Final shapes - SHAP: {shap_values_trimmed.shape}, Data: {X_sample_trimmed.shape}, Names: {len(feature_names_trimmed)}")

        # Create the plot using matplotlib directly to avoid SHAP's internal sorting issues
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean absolute SHAP values for sorting
        mean_abs_shap = np.abs(shap_values_trimmed).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)
        
        # Plot top features
        top_n = min(20, n_common)
        top_indices = sorted_indices[-top_n:]
        
        for idx in top_indices:
            feature_shap_values = shap_values_trimmed[:, idx]
            feature_data_values = X_sample_trimmed[:, idx]
            
            # Create scatter plot for this feature
            y_pos = np.where(top_indices == idx)[0][0]
            
            # Add jitter to y-axis for better visualization
            y_jitter = np.random.normal(y_pos, 0.1, size=len(feature_shap_values))
            
            scatter = ax.scatter(
                feature_shap_values,
                y_jitter,
                c=feature_data_values,
                cmap='coolwarm',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        
        # Set y-axis labels
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names_trimmed[i] for i in top_indices])
        ax.set_xlabel('SHAP value (impact on model output)')
        ax.set_title(f'SHAP Summary Plot{class_label}')
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Feature value', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved to {output_path}")
    
    except Exception as e:
        import traceback
        print(f"Error generating SHAP summary plot: {e}")
        print(traceback.format_exc())
        plt.close('all')  # Clean up any partial plots


def get_class_order(model, label_encoder=None):
    """Return the list of class labels in the order corresponding to predict_proba columns."""
    if not hasattr(model, "classes_"):
        return None
    classes = model.classes_
    if label_encoder is not None:
        classes = label_encoder.inverse_transform(classes)
    return list(classes)


def compute_one_vs_rest_roc(model, X_test, y_test, label_encoder=None):
    """Compute one-vs-rest ROC curves and AUCs for every class the model knows.

    Returns dict mapping class label -> {"fpr": ..., "tpr": ..., "auc": ..., "support": int}.
    Returns None if the model lacks predict_proba.
    """
    if not hasattr(model, "predict_proba"):
        return None

    proba = model.predict_proba(X_test)
    class_order = get_class_order(model, label_encoder=label_encoder)
    if class_order is None or proba.shape[1] != len(class_order):
        return None

    y_arr = np.asarray(y_test)
    results = {}
    for col_idx, cls in enumerate(class_order):
        y_binary = (y_arr == cls).astype(int)
        support = int(y_binary.sum())
        if support == 0 or support == len(y_binary):
            # ROC undefined when one of the two binary classes is empty
            continue
        fpr, tpr, _ = roc_curve(y_binary, proba[:, col_idx])
        results[str(cls)] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": float(auc(fpr, tpr)),
            "support": support,
        }
    return results


def save_per_model_roc_plot(roc_data, output_path, title):
    """Plot one-vs-rest ROC curves (all classes) for a single model on one figure."""
    if not roc_data:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in sorted(roc_data.keys()):
        d = roc_data[cls]
        ax.plot(d["fpr"], d["tpr"], lw=2, label=f"{cls} (AUC = {d['auc']:.3f}, n = {d['support']})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC plot saved to {output_path}")


def save_roc_comparison_plot(roc_lgb, roc_xgb, output_path, label_lgb, label_xgb):
    """Compare two models' ROC curves side-by-side, one subplot per class."""
    if not roc_lgb or not roc_xgb:
        print("Skipping ROC comparison plot (missing data for one or both models).")
        return

    classes = sorted(set(roc_lgb.keys()) & set(roc_xgb.keys()))
    if not classes:
        print("Skipping ROC comparison plot (no overlapping classes).")
        return

    n = len(classes)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 5 * rows), squeeze=False)
    for i, cls in enumerate(classes):
        ax = axes[i // cols][i % cols]
        d_lgb = roc_lgb[cls]
        d_xgb = roc_xgb[cls]
        ax.plot(d_lgb["fpr"], d_lgb["tpr"], lw=2,
                label=f"{label_lgb} (AUC = {d_lgb['auc']:.3f})", color="#1f77b4")
        ax.plot(d_xgb["fpr"], d_xgb["tpr"], lw=2,
                label=f"{label_xgb} (AUC = {d_xgb['auc']:.3f})", color="#d62728")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Class: {cls} (n = {d_lgb['support']})")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplot
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    plt.suptitle(f"ROC Comparison: {label_lgb} vs. {label_xgb} (one-vs-rest)", y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC comparison plot saved to {output_path}")


def compute_macro_specificity_sensitivity(y_true, y_pred, labels):
    """Compute macro-averaged sensitivity and specificity using one-vs-rest.

    Sensitivity for class c = TP_c / (TP_c + FN_c) = recall for c.
    Specificity for class c = TN_c / (TN_c + FP_c).
    Returned values are simple averages across classes (macro).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    sens, spec = [], []
    per_class = {}
    for i, cls in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens_c = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec_c = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        sens.append(sens_c)
        spec.append(spec_c)
        per_class[str(cls)] = {"sensitivity": sens_c, "specificity": spec_c}
    macro_sens = float(np.nanmean(sens))
    macro_spec = float(np.nanmean(spec))
    return {"macro_sensitivity": macro_sens, "macro_specificity": macro_spec, "per_class": per_class}


def compute_paper_style_metrics(y_true, y_pred, roc_data, labels):
    """Aggregate the per-model metric row used in the paper-style comparison."""
    from sklearn.metrics import precision_recall_fscore_support

    accuracy = accuracy_score(y_true, y_pred)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    sens_spec = compute_macro_specificity_sensitivity(y_true, y_pred, labels)

    auc_macro = float("nan")
    auc_weighted = float("nan")
    if roc_data:
        aucs = np.array([d["auc"] for d in roc_data.values()])
        supports = np.array([d["support"] for d in roc_data.values()])
        if len(aucs) > 0:
            auc_macro = float(aucs.mean())
            auc_weighted = float(np.average(aucs, weights=supports)) if supports.sum() > 0 else float("nan")

    return {
        "accuracy": accuracy,
        "precision_macro": precision_m,
        "recall_macro": recall_m,
        "f1_macro": f1_m,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "sensitivity_macro": sens_spec["macro_sensitivity"],
        "specificity_macro": sens_spec["macro_specificity"],
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
    }


def save_paper_style_bar_chart(metrics_by_model, output_path):
    """Grouped bar chart comparing models across Accuracy / Precision / Recall / F1 (macro)."""
    if not metrics_by_model:
        return
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    models = list(metrics_by_model.keys())
    n_models = len(models)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    palette = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        values = [metrics_by_model[model][k] for k in metric_keys]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model, color=palette[i % len(palette)])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score (macro-averaged for multiclass)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Machine Learning Model Performance Evaluation")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Paper-style bar chart saved to {output_path}")


def save_paper_style_roc_panel(roc_by_model, output_path):
    """Multi-panel ROC figure with one panel per model (each panel shows per-class ROC curves)."""
    models = [m for m in roc_by_model if roc_by_model[m]]
    if not models:
        return

    n = len(models)
    cols = n
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5.5), squeeze=False)
    panel_letters = ["A", "B", "C", "D"]

    for i, model in enumerate(models):
        ax = axes[0][i]
        roc_data = roc_by_model[model]
        for cls in sorted(roc_data.keys()):
            d = roc_data[cls]
            ax.plot(d["fpr"], d["tpr"], lw=2, label=f"{cls} (AUC = {d['auc']:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"({panel_letters[i]}) ROC — {model}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Receiver Operating Characteristic Curves (one-vs-rest) per Model", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Paper-style ROC panel saved to {output_path}")


def build_roc_summary_rows(label, roc_data, total_support):
    """Build per-class + macro/weighted summary rows for a single model."""
    if not roc_data:
        return []
    rows = []
    for cls in sorted(roc_data.keys()):
        d = roc_data[cls]
        rows.append({
            "model": label,
            "class": cls,
            "auc": round(d["auc"], 4),
            "support": d["support"],
        })
    aucs = np.array([d["auc"] for d in roc_data.values()])
    supports = np.array([d["support"] for d in roc_data.values()])
    macro = float(aucs.mean())
    weighted = float(np.average(aucs, weights=supports)) if supports.sum() > 0 else float("nan")
    rows.append({"model": label, "class": "MACRO", "auc": round(macro, 4), "support": total_support})
    rows.append({"model": label, "class": "WEIGHTED", "auc": round(weighted, 4), "support": total_support})
    return rows


def main():
    X, y = load_dataset()

    preproc_path = os.path.join(OUTPUT_DIR, PREPROCESSOR_FILE)
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Missing preprocessing pipeline at {preproc_path}")
    preproc = joblib.load(preproc_path)

    feature_names = extract_feature_names(preproc)
    transformed_preview = preproc.transform(X.head(1))
    expected_dim = transformed_preview.shape[1]
    if not feature_names or len(feature_names) != expected_dim:
        if feature_names:
            print(
                f"Warning: ColumnTransformer reported {len(feature_names)} names but "
                f"transformed data has {expected_dim} columns. Regenerating generic names."
            )
        feature_names = [f"feature_{i}" for i in range(expected_dim)]

    X_test_array, y_test = prepare_test_split(preproc, X, y)
    if len(feature_names) != X_test_array.shape[1]:
        print(
            f"Warning: Feature name count ({len(feature_names)}) does not match "
            f"data columns ({X_test_array.shape[1]}). Adjusting names."
        )
        if len(feature_names) < X_test_array.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X_test_array.shape[1])]
        else:
            feature_names = feature_names[: X_test_array.shape[1]]
    X_test = pd.DataFrame(X_test_array, columns=feature_names)

    xgb_encoder_path = os.path.join(OUTPUT_DIR, XGB_LABEL_ENCODER_FILE)
    xgb_label_encoder = joblib.load(xgb_encoder_path) if os.path.exists(xgb_encoder_path) else None

    evaluation_results = {}
    for label, filename in MODEL_FILES.items():
        model_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(model_path):
            print(f"Skipping {label} (missing artifact at {model_path}).")
            continue
        model = joblib.load(model_path)
        encoder = xgb_label_encoder if "XGBoost" in label else None
        if "XGBoost" in label and encoder is None:
            print(f"Skipping {label} (XGB label encoder missing at {xgb_encoder_path}).")
            continue
        y_pred = evaluate_model(model, X_test, y_test, label, label_encoder=encoder)
        evaluation_results[label] = {"model": model, "y_pred": y_pred, "encoder": encoder}

    if not evaluation_results:
        raise RuntimeError("No saved models were found for evaluation.")

    primary_label = next((lbl for lbl in MODEL_PRIORITY if lbl in evaluation_results), next(iter(evaluation_results)))
    primary_result = evaluation_results[primary_label]
    print(f"\nGenerating visualizations with: {primary_label}")

    cm_path = os.path.join(OUTPUT_DIR, CONFUSION_MATRIX_IMG)
    save_confusion_matrix_plot(y_test, primary_result["y_pred"], cm_path, f"Confusion Matrix - {primary_label}")

    fi_csv_path = os.path.join(OUTPUT_DIR, FEATURE_IMPORTANCE_CSV)
    fi_img_path = os.path.join(OUTPUT_DIR, FEATURE_IMPORTANCE_IMG)
    save_feature_importance_outputs(primary_result["model"], feature_names, fi_csv_path, fi_img_path)

    shap_path = os.path.join(OUTPUT_DIR, SHAP_SUMMARY_IMG)
    save_shap_summary_plot(primary_result["model"], X_test, feature_names, shap_path)

    # ----- ROC / AUC analysis (one-vs-rest, all models) -----
    print("\n===== ROC / AUC analysis =====")
    summary_rows = []
    roc_by_label = {}
    total_support = len(y_test)
    for label, result in evaluation_results.items():
        roc_data = compute_one_vs_rest_roc(
            result["model"], X_test, y_test, label_encoder=result.get("encoder")
        )
        if not roc_data:
            print(f"Skipping {label} (no predict_proba or class mismatch).")
            continue
        roc_by_label[label] = roc_data

        # Per-model ROC plot
        slug = ROC_PLOT_SLUGS.get(label, label.lower().replace(" ", "_"))
        plot_path = os.path.join(OUTPUT_DIR, f"roc_curves_{slug}.png")
        save_per_model_roc_plot(roc_data, plot_path, f"ROC Curves (one-vs-rest) — {label}")

        # Summary print
        per_class = {cls: round(d["auc"], 4) for cls, d in roc_data.items()}
        aucs = np.array([d["auc"] for d in roc_data.values()])
        supports = np.array([d["support"] for d in roc_data.values()])
        macro = aucs.mean()
        weighted = np.average(aucs, weights=supports) if supports.sum() > 0 else float("nan")
        print(f"{label}: per-class AUC = {per_class} | macro = {macro:.4f} | weighted = {weighted:.4f}")

        summary_rows.extend(build_roc_summary_rows(label, roc_data, total_support))

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(OUTPUT_DIR, ROC_SUMMARY_CSV)
        summary_df.to_csv(summary_csv, index=False)
        print(f"ROC AUC summary saved to {summary_csv}")

    # Tuned LightGBM vs Tuned XGBoost head-to-head ROC comparison
    if "Tuned LightGBM" in roc_by_label and "Tuned XGBoost" in roc_by_label:
        comparison_path = os.path.join(OUTPUT_DIR, ROC_COMPARISON_IMG)
        save_roc_comparison_plot(
            roc_by_label["Tuned LightGBM"],
            roc_by_label["Tuned XGBoost"],
            comparison_path,
            "Tuned LightGBM",
            "Tuned XGBoost",
        )

    # ----- Paper-style multi-model comparison (LR / LightGBM / XGBoost) -----
    print("\n===== Paper-style multi-model comparison =====")
    class_labels = sorted(pd.unique(np.asarray(y_test)))
    paper_metrics = {}
    paper_roc_by_model = {}
    for source_label in PAPER_STYLE_MODELS:
        if source_label not in evaluation_results:
            print(f"Skipping {source_label} (artifact not loaded).")
            continue
        display = PAPER_STYLE_DISPLAY.get(source_label, source_label)
        result = evaluation_results[source_label]
        roc_data = roc_by_label.get(source_label)
        metrics = compute_paper_style_metrics(y_test, result["y_pred"], roc_data, class_labels)
        paper_metrics[display] = metrics
        if roc_data:
            paper_roc_by_model[display] = roc_data

        print(
            f"{display}: acc={metrics['accuracy']:.3f} | "
            f"P(macro)={metrics['precision_macro']:.3f} | "
            f"R(macro)={metrics['recall_macro']:.3f} | "
            f"F1(macro)={metrics['f1_macro']:.3f} | "
            f"Sens(macro)={metrics['sensitivity_macro']:.3f} | "
            f"Spec(macro)={metrics['specificity_macro']:.3f} | "
            f"AUC(macro)={metrics['auc_macro']:.3f}"
        )

    if paper_metrics:
        rows = []
        for model, m in paper_metrics.items():
            row = {"model": model}
            for k, v in m.items():
                row[k] = round(v, 4) if isinstance(v, float) else v
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, PAPER_METRICS_CSV), index=False)
        print(f"Paper-style metrics CSV saved to {os.path.join(OUTPUT_DIR, PAPER_METRICS_CSV)}")

        save_paper_style_bar_chart(paper_metrics, os.path.join(OUTPUT_DIR, PAPER_METRICS_BAR_IMG))
        save_paper_style_roc_panel(paper_roc_by_model, os.path.join(OUTPUT_DIR, PAPER_ROC_PANEL_IMG))


if __name__ == "__main__":
    main()

