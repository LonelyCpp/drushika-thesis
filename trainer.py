"""
trainer.py

Usage:
  - Update DATA_PATH and TARGET_COLUMN below to match your dataset.
  - Then run: python trainer.py
Outputs will be written to OUTPUT_DIR (created if missing).
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

# ********** USER SETTINGS **********
DATA_PATH = "./data.csv"
TARGET_COLUMN = "Risk"  # <-- change this to the actual label column in your sheet
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_ITER_SEARCH = 20
CV_FOLDS = 4
OUTPUT_DIR = "./model_outputs"
# ***********************************

os.makedirs(OUTPUT_DIR, exist_ok=True)

# read tabular data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"DATA_PATH '{DATA_PATH}' does not exist.")

read_kwargs = {}
if DATA_PATH.lower().endswith((".xlsx", ".xls")):
    # maintain compatibility with legacy Excel files
    sheet_name = None  # default to first sheet
    df = pd.read_excel(DATA_PATH, sheet_name=sheet_name, **read_kwargs)
else:
    df = pd.read_csv(DATA_PATH, **read_kwargs)

# normalize column names to strings (strip spaces)
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
    y = y.replace({"N": "NO", "": np.nan})
# if numeric but integer-like, convert to int
if pd.api.types.is_float_dtype(y) and y.dropna().apply(float.is_integer).all():
    y = y.astype('Int64')

# drop rows with missing targets
if y.isna().any():
    missing_idx = y[y.isna()].index
    print(f"Dropping {len(missing_idx)} rows with missing target.")
    X = X.drop(index=missing_idx)
    y = y.drop(index=missing_idx)

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
value_counts = y.value_counts()
if y.nunique() <= 10 and value_counts.min() >= 2:
    stratify_arg = y
else:
    stratify_arg = None
    if y.nunique() <= 10 and value_counts.min() < 2:
        print("Skipping stratified split because minimum class count < 2.")
X_train, X_test, y_train, y_test = train_test_split(X_pre, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_arg)

# Baseline: Logistic Regression (for clinical interpretability)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression report:\n", classification_report(y_test, y_pred_lr))
joblib.dump(lr, os.path.join(OUTPUT_DIR, "logistic_regression.joblib"))

# LightGBM training and tuning
LGB = True
lgb_import_error = None
try:
    import lightgbm as lgb
except ModuleNotFoundError as exc:
    LGB = False
    lgb_import_error = f"{exc.__class__.__name__}: {exc}"
except Exception as exc:  # surface unexpected import issues
    raise RuntimeError("LightGBM is installed but failed to import.") from exc
finally:
    if not LGB:
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
    reason = lgb_import_error or "unknown import failure"
    print(f"LightGBM not available ({reason}). Skipped hyperparameter tuning.")

# XGBoost training and tuning (for head-to-head comparison with LightGBM)
XGB = True
xgb_import_error = None
try:
    import xgboost as xgb
except ModuleNotFoundError as exc:
    XGB = False
    xgb_import_error = f"{exc.__class__.__name__}: {exc}"
except Exception as exc:
    raise RuntimeError("xgboost is installed but failed to import.") from exc

if XGB:
    # XGBoost requires integer-encoded labels for multi-class. Persist the encoder
    # alongside the model so evaluation can decode predictions back to strings.
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "xgb_label_encoder.joblib"))

    xgb_baseline = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=4,
        eval_metric="mlogloss",
        tree_method="hist",
    )
    xgb_baseline.fit(X_train, y_train_enc)
    y_pred_xgb_base_enc = xgb_baseline.predict(X_test)
    y_pred_xgb_base = label_encoder.inverse_transform(y_pred_xgb_base_enc)
    print("Baseline XGBoost report:\n", classification_report(y_test, y_pred_xgb_base))
    print("Confusion matrix (XGBoost baseline):\n", confusion_matrix(y_test, y_pred_xgb_base))
    joblib.dump(xgb_baseline, os.path.join(OUTPUT_DIR, "xgb_baseline_model.joblib"))

    xgb_param_dist = {
        "max_depth": [3, 5, 7, 9],
        "n_estimators": [50, 100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [0, 0.01, 0.1],
    }
    xgb_scoring = "f1" if y.nunique() == 2 else "f1_weighted"
    xgb_rs = RandomizedSearchCV(
        xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=4,
            eval_metric="mlogloss",
            tree_method="hist",
        ),
        param_distributions=xgb_param_dist,
        n_iter=N_ITER_SEARCH,
        scoring=xgb_scoring,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    xgb_rs.fit(X_train, y_train_enc)
    print("Best XGBoost params:", xgb_rs.best_params_)
    xgb_best = xgb_rs.best_estimator_
    joblib.dump(xgb_best, os.path.join(OUTPUT_DIR, "xgb_tuned_model.joblib"))
    y_pred_xgb_best = label_encoder.inverse_transform(xgb_best.predict(X_test))
    print("Tuned XGBoost report:\n", classification_report(y_test, y_pred_xgb_best))
    print("Confusion matrix (XGBoost tuned):\n", confusion_matrix(y_test, y_pred_xgb_best))
else:
    reason = xgb_import_error or "unknown import failure"
    print(f"XGBoost not available ({reason}). Skipping XGBoost training.")

# If binary label, save ROC AUC
if y.nunique() == 2:
    if LGB:
        proba = best_model.predict_proba(X_test)[:, 1]
    else:
        proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(pd.Series(y_test).astype(float), proba.astype(float))
    print("ROC AUC:", auc)

print("All outputs saved to:", OUTPUT_DIR)
