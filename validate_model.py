"""
validate_model.py

Run the saved tuned LightGBM model against validate.csv (a held-out set with
no header) and report accuracy, per-class metrics, and confusion matrix.

Usage:
  python3 validate_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

VALIDATE_PATH = "./validate.csv"
OUTPUT_DIR = "./model_outputs"
PREPROCESSOR_FILE = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
MODEL_FILE = os.path.join(OUTPUT_DIR, "lgb_tuned_model.joblib")

COLUMN_NAMES = ["Age", "Sex", "Diet Score", "Risk"]
LABEL_MAP = {"H": "HIGH", "M": "MODERATE", "L": "LOW", "NO": "NO", "N": "NO"}


def load_validate():
    df = pd.read_csv(VALIDATE_PATH, header=None, names=COLUMN_NAMES)
    df["Risk"] = df["Risk"].astype(str).str.strip().map(LABEL_MAP)
    if df["Risk"].isna().any():
        bad = df[df["Risk"].isna()]
        raise ValueError(f"Unrecognized labels in validate.csv:\n{bad}")
    return df.drop(columns=["Risk"]), df["Risk"]


def main():
    X, y_true = load_validate()
    print(f"Loaded {len(X)} validation rows")
    print("Label distribution:\n", y_true.value_counts())

    preproc = joblib.load(PREPROCESSOR_FILE)
    model = joblib.load(MODEL_FILE)

    X_pre = preproc.transform(X)
    y_pred = model.predict(X_pre)

    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    labels = sorted(set(np.concatenate([y_true.values, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Labels:", labels)
    print(cm)


if __name__ == "__main__":
    main()
