## **Context Document 

**Project:** Machine Learning model for predicting dental caries risk in children based on 7-day diet diary data and derived nutrition/behavioral features.

**Dataset:**

* `~754 rows × ~13 columns` (Excel sheet)
* Includes demographic information, diet diary derived variables, oral hygiene habits, and a **target column** indicating **caries risk category** (e.g., `High`, `Moderate`, `Low`).
* The goal is to predict this risk category based on the other features.

**Objective:**
Train and tune a **LightGBM classifier** to predict caries risk and compare performance against a baseline **Logistic Regression** model.
Evaluate metrics including **accuracy, precision, recall, F1 score, confusion matrix, and ROC-AUC** (if binary). Generate feature importance and SHAP interpretability outputs.

**Method Overview:**

1. Load dataset from Excel.
2. Identify `TARGET_COLUMN` (label for prediction).
3. Split data into train/test (70/30), stratified.
4. Preprocess features (median imputation for numeric, ordinal encoding for categoricals).
5. Train baseline **LogisticRegression** and baseline **LightGBMClassifier**.
6. Perform hyperparameter tuning using **RandomizedSearchCV** with 4-fold CV.
7. Evaluate tuned model using classification report, confusion matrix, ROC-AUC.
8. Generate feature importances and optionally SHAP summary plot.
9. Save trained model and preprocessing pipeline via joblib.

**LightGBM tuning parameters search space example:**

```
{
  "num_leaves": [15, 31, 63, 127],
  "n_estimators": [50, 100, 200, 400],
  "learning_rate": [0.01, 0.05, 0.1],
  "min_child_samples": [5, 10, 20, 50],
  "subsample": [0.6, 0.8, 1.0],
  "colsample_bytree": [0.6, 0.8, 1.0],
  "reg_alpha": [0, 0.01, 0.1],
  "reg_lambda": [0, 0.01, 0.1]
}
```

**Required Python Libraries:**

```
pip install pandas numpy scikit-learn lightgbm joblib matplotlib shap seaborn openpyxl
```

**Output Deliverables:**

* Classification metrics table comparing models
* Confusion matrix figure
* ROC curve (if binary classification)
* Feature importance (CSV + plot)
* SHAP global summary plot (optional)
* Saved model files (`model.joblib`, `preprocessor.joblib`)

**Primary Questions to Ask Codex/GPT During Development:**

* Help identifying the correct `TARGET_COLUMN`
* Assistance cleaning or engineering features from raw diary inputs
* Code fixes for training/tuning errors
* Generating thesis-style Results and Discussion text from metrics

---

## Usage instructions for Codex/GPT

When interacting with the model, begin message like:

> “Use the following context for all responses unless I say otherwise: [paste context document]. I will now ask questions related to training LightGBM on this dataset…”

--- 

72-96 – no caries risk
64-72 – low caries risk
56 -64 moderate caries risk 
56 or less high caries risk

---

## Operational Brief (for future runs)

### Files
- `data.csv` — training dataset (header row: `Age,Sex,Diet Score,Risk`). Risk values: `HIGH`, `MODERATE`, `LOW`, `NO`.
- `validate.csv` — external validation set (no header, same column order). Labels are abbreviated: `H`, `M`, `L`, `NO`.
- `trainer.py` — fits preprocessor, baseline LR, baseline LightGBM, and tuned LightGBM via RandomizedSearchCV. Writes models + preprocessor to `model_outputs/`.
- `evaluate_models.py` — loads saved artifacts, recreates the deterministic 70/30 split, and writes `confusion_matrix.png`, `feature_importance.png`, `feature_importances.csv`, and `shap_summary.png`. Does **not** write `results.md` — that file is maintained by hand.
- `validate_model.py` — runs the saved tuned model against `validate.csv` and prints accuracy / classification report / confusion matrix.

### Standard refresh workflow (after data.csv changes)
```bash
python3 trainer.py && python3 evaluate_models.py 2>&1 | tee model_outputs/eval_run.log
python3 validate_model.py 2>&1 | tee model_outputs/validate_run.log
```
Then update `README.md` and `model_outputs/results.md` from the two log files.

### Key settings (must stay consistent across trainer.py and evaluate_models.py)
- `RANDOM_STATE = 42`
- `TEST_SIZE = 0.30` (stratified)
- `TARGET_COLUMN = "Risk"`
- 4-fold CV, 20 RandomizedSearchCV iterations
- Scoring: `f1_weighted` (multi-class)

### Data quirks to remember
- `validate.csv` has **no header row**, abbreviated labels (`H/M/L/NO`), and label-rubric inconsistencies near the HIGH/MOD/LOW boundaries. The 80% external accuracy reported in `README.md` reflects this; a relabeled clean validation set is the planned fix.
- Negative Diet Scores in `validate.csv` were originally data-entry errors. Per the clinician's instruction (2026-04-26), negatives are flipped to positive (HIGH risk by rubric).
- Training data has its own rubric overlaps (e.g., HIGH labels on DS up to 85, NO down to 64). Don't assume strict rubric compliance when reasoning about predictions.

### Output artifacts (under `model_outputs/`)
- `baseline_model.joblib`, `lgb_tuned_model.joblib`, `preprocessor.joblib` — saved sklearn artifacts
- `feature_importances.csv`, `feature_importance.png`, `confusion_matrix.png`, `shap_summary.png`
- `results.md` — hand-maintained metrics summary
- `eval_run.log`, `validate_run.log` — most recent run captures (regenerated via `tee`)