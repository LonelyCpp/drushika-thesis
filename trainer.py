"""
train_lightgbm_pipeline.py

Usage:
  - Edit TARGET_COLUMN below to match the column name in your Excel sheet that is the label.
  - Then run: python train_lightgbm_pipeline.py
Outputs will be written to /mnt/data/ (or current working dir) by default.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

# ********** USER SETTINGS **********
EXCEL_PATH = "./data.xlsx"
SHEET_NAME = None  # None -> first sheet
TARGET_COLUMN = "Unnamed: 12"  # <-- change this to the actual label column in your sheet
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_ITER_SEARCH = 20
CV_FOLDS = 4
OUTPUT_DIR = "./model_outputs"
# ***********************************

os.makedirs(OUTPUT_DIR, exist_ok=True)

# read sheet
xls = pd.ExcelFile(EXCEL_PATH)
sheet = SHEET_NAME if SHEET_NAME else xls.sheet_names[0]
df = xls.parse(sheet)
# normalize column names to strings
df.columns = [str(c).strip() for c in df.columns]

# quick inspect
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Sample rows:\n", df.head())

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"TARGET_COLUMN '{TARGET_COLUMN}' not found. Columns: {df.columns.tolist()}")

# split X/y
X = df.drop(columns=[TARGET_COLUMN]).copy()
y = df[TARGET_COLUMN].copy()

# basic cleanup for y
if y.dtype == object:
    y = y.astype(str).str.strip()
# if numeric but integer-like, convert to int
if pd.api.types.is_float_dtype(y) and y.dropna().apply(float.is_integer).all():
    y = y.astype('Int64')

print("Target value counts:\n", y.value_counts(dropna=False))

# column typing
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print("Numeric cols:", num_cols)
print("Categorical cols:", cat_cols)

# Preprocessing pipelines
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
preproc = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
], remainder='drop')

# transform
X_pre = preproc.fit_transform(X)
feature_names = num_cols + cat_cols

# train-test split (stratify when label is categorical with few classes)
stratify_arg = y if y.nunique() <= 10 else None
X_train, X_test, y_train, y_test = train_test_split(X_pre, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_arg)

# Baseline: Logistic Regression (for clinical interpretability)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression report:\n", classification_report(y_test, y_pred_lr))

# LightGBM training and tuning
try:
    import lightgbm as lgb
    LGB = True
except Exception:
    LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

if LGB:
    model = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=4)
else:
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

# baseline LightGBM
model.fit(X_train, y_train)
y_pred_base = model.predict(X_test)
print("Baseline model report:\n", classification_report(y_test, y_pred_base))
cm = confusion_matrix(y_test, y_pred_base)
print("Confusion matrix (baseline):\n", cm)

# Save baseline artifact
joblib.dump(preproc, os.path.join(OUTPUT_DIR, "preprocessor.joblib"))
joblib.dump(model, os.path.join(OUTPUT_DIR, "baseline_model.joblib"))

# Hyperparameter search (only if LightGBM available)
best_model = model
if LGB:
    param_dist = {
        "num_leaves": [15, 31, 63, 127],
        "n_estimators": [50, 100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_samples": [5, 10, 20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [0, 0.01, 0.1]
    }
    scoring = "f1" if y.nunique() == 2 else "f1_weighted"
    rs = RandomizedSearchCV(lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=4),
                             param_distributions=param_dist,
                             n_iter=N_ITER_SEARCH,
                             scoring=scoring,
                             cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                             random_state=RANDOM_STATE,
                             n_jobs=1,
                             verbose=1)
    rs.fit(X_train, y_train)
    print("Best params:", rs.best_params_)
    best_model = rs.best_estimator_
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "lgb_tuned_model.joblib"))
    y_pred_best = best_model.predict(X_test)
    print("Tuned model report:\n", classification_report(y_test, y_pred_best))
    cm_best = confusion_matrix(y_test, y_pred_best)
    print("Confusion matrix (tuned):\n", cm_best)

    # Feature importance
    fi = best_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)
    print("Feature importances saved to", os.path.join(OUTPUT_DIR, "feature_importances.csv"))
else:
    print("LightGBM not installed. Skipped hyperparameter tuning.")

# If binary label, save ROC AUC
if y.nunique() == 2:
    if LGB:
        proba = best_model.predict_proba(X_test)[:, 1]
    else:
        proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(pd.Series(y_test).astype(float), proba.astype(float))
    print("ROC AUC:", auc)

print("All outputs saved to:", OUTPUT_DIR)
