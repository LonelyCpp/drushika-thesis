# Machine Learning for Dental Caries Risk Prediction in Children

**Student:** Dr. Drushika Dinesh  
**Project Type:** PG Student Dissertation Research

---

## 📋 Project Overview

This research project develops and evaluates a machine learning-based algorithm to predict dental caries risk in children using dietary intake data collected from 7-day diet diaries. The primary objective is to create an automated system that can classify children into caries risk categories (High, Moderate, Low, or No Risk) based on their dietary patterns, enabling early intervention and personalized preventive dental care.

### Research Objectives

1. **To assess dental caries risk** using 7-day diet diary records and calculate Dental Health Diet Scores
2. **To develop a machine learning algorithm** that predicts caries risk potential from dietary intake patterns in children

---

## 🦷 Dental Health Diet Score Methodology

### Data Collection Process

For each child participant, a comprehensive 7-day diet diary was recorded, capturing:

- **Timing** of meals and snacks consumed
- **Amount ingested** (measured in household measures)
- **Food preparation method**
- **Added sugar content** (teaspoons of sugar added)

Foods containing added sugar or concentrated natural sweets (honey, raisins, figs, etc.) were identified and classified into appropriate food groups for risk assessment.

### Risk Classification Scale

The Dental Diet Score categorizes children into four risk levels:

| Score Range | Classification | Caries Risk Level |
|-------------|----------------|-------------------|
| 72-96       | Excellent      | **No Risk**       |
| 64-72       | Adequate       | **Low Risk**      |
| 56-64       | Barely Adequate| **Moderate Risk** |
| ≤56         | Not Adequate   | **High Risk**     |

The obtained diet diary scores were compared with DMFT/deft index scores to validate caries risk assessment.

---

## 🤖 Machine Learning Approach

### Dataset Characteristics

- **Total Samples:** 754 children
- **Features:** 12 predictive variables (demographic, dietary metrics, oral hygiene habits)
- **Target Variable:** Caries risk category (HIGH, MODERATE, LOW, NO)
- **Data Split:** 70% training (528 samples) / 30% testing (226 samples)
- **Split Strategy:** Stratified random sampling to maintain class distribution

### Models Implemented

#### 1. Baseline Model: Logistic Regression
A traditional statistical classifier was implemented as a baseline for comparison, providing clinical interpretability through coefficient analysis.

#### 2. Primary Model: LightGBM (Gradient Boosting)
A state-of-the-art gradient boosting framework optimized for:
- Handling complex non-linear relationships in dietary patterns
- Managing class imbalance across risk categories
- Providing feature importance rankings
- Computational efficiency with large datasets

### Hyperparameter Optimization

The LightGBM model underwent extensive tuning using **RandomizedSearchCV** with 4-fold stratified cross-validation:

**Search Space:**
```python
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

**Optimization Metric:** Weighted F1-score (to account for class imbalance)

---

## 📊 Results

### Model Performance on Test Set

| Metric                | Score              |
| --------------------- | ------------------ |
| **Accuracy**          | **91.6%**          |
| **Weighted F1-score** | **0.90**           |

### Per-Class Performance Metrics

| Risk Category | Precision | Recall | F1-score | Test Samples |
|---------------|-----------|--------|----------|--------------|
| **HIGH**      | 0.95      | 0.98   | **0.97** | 177          |
| **LOW**       | 0.78      | 0.90   | **0.84** | 20           |
| **MODERATE**  | 0.50      | 0.14   | 0.22     | 14           |
| **NO**        | 0.82      | 0.88   | **0.85** | 16           |

### Confusion Matrix

```
               Predicted Risk Category
               HIGH  LOW  MODERATE  NO
Actual HIGH    174    0      1      2
Actual LOW       0   18      1      1
Actual MOD       9    3      2      0
Actual NO        0    2      0     14
```

### Key Findings

✅ **Excellent HIGH-risk detection:** The model achieved 98% recall for HIGH-risk children, meaning it successfully identifies nearly all children who are actually at high risk for caries. This is clinically critical for early intervention.

✅ **Strong overall performance:** With 91.6% accuracy, the model demonstrates robust predictive capability across most risk categories.

✅ **Reliable LOW and NO risk classification:** F1-scores of 0.84-0.85 indicate balanced precision and recall for these categories.

⚠️ **Moderate-risk limitation:** Performance on MODERATE risk cases (F1 = 0.22) was constrained by limited representation (n=14 samples). Most misclassifications assigned moderate cases to the HIGH category, suggesting overlapping dietary patterns between these groups.

### Feature Importance

The three most influential predictors of caries risk were:

| Feature      | Importance Score |
|--------------|------------------|
| Diet Score   | 3019            |
| Age          | 197             |
| Sex          | 127             |

The **Diet Score** was by far the dominant predictor, accounting for approximately 90% of the model's decision-making, which validates the clinical hypothesis that dietary patterns are the primary driver of caries risk in children.

---

## 📈 Visualizations

### 1. Confusion Matrix
![Confusion Matrix](model_outputs/confusion_matrix.png)

Visual representation of the model's classification performance across all four risk categories.

### 2. Feature Importance
![Feature Importance](model_outputs/feature_importance.png)

Bar chart showing the relative contribution of each feature to the model's predictions.

### 3. SHAP Summary Plot
![SHAP Analysis](model_outputs/shap_summary.png)

SHAP (SHapley Additive exPlanations) values illustrating how each feature impacts individual predictions, with color indicating feature values and horizontal position showing positive or negative influence on HIGH-risk classification.

---

## 🛠️ Technical Implementation

### Prerequisites

- Python 3.8 or higher
- Required packages listed in `requirements.txt`

### Installation

```bash
# Clone or download the repository
cd drushika

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
drushika/
├── data.csv                    # Dataset (7-day diet diary records)
├── trainer.py                  # Model training pipeline
├── evaluate_models.py          # Model evaluation and visualization
├── requirements.txt            # Python dependencies
├── context.md                  # Project methodology documentation
└── model_outputs/              # Generated artifacts
    ├── baseline_model.joblib       # Trained Logistic Regression model
    ├── lgb_tuned_model.joblib      # Optimized LightGBM model
    ├── preprocessor.joblib         # Data preprocessing pipeline
    ├── confusion_matrix.png        # Classification confusion matrix
    ├── feature_importance.png      # Feature importance visualization
    ├── feature_importances.csv     # Feature importance rankings
    ├── shap_summary.png            # SHAP interpretability plot
    └── results.md                  # Detailed performance metrics
```

### Usage

#### Training the Model

```bash
python trainer.py
```

This script will:
1. Load and preprocess the diet diary dataset
2. Split data into training (70%) and test (30%) sets
3. Train baseline Logistic Regression model
4. Train baseline LightGBM model
5. Perform hyperparameter tuning with cross-validation
6. Save all trained models to `model_outputs/`

#### Evaluating the Model

```bash
python evaluate_models.py
```

This script will:
1. Load the saved models and preprocessing pipeline
2. Generate predictions on the test set
3. Calculate performance metrics (accuracy, precision, recall, F1)
4. Create visualizations (confusion matrix, feature importance, SHAP)
5. Save all outputs to `model_outputs/`

---

## 📝 Clinical Interpretation

### Implications for Dental Practice

The tuned LightGBM model demonstrates **clinically viable performance** for automated caries risk screening in pediatric dentistry:

1. **High Sensitivity for At-Risk Children:** With 98% recall for HIGH-risk cases, the model minimizes false negatives—ensuring that children who need urgent dietary intervention are identified.

2. **Dietary Focus Validated:** The overwhelming importance of the Diet Score feature (90% contribution) confirms that dietary patterns captured in 7-day diaries are highly predictive of caries risk, supporting evidence-based dietary counseling.

3. **Practical Deployment:** The model can be integrated into dental clinics or school health programs to:
   - Automatically score diet diaries
   - Flag high-risk children for immediate follow-up
   - Provide personalized dietary recommendations
   - Track risk changes over time with repeat assessments

### Limitations and Recommendations

**Class Imbalance:** The MODERATE risk category was underrepresented (n=14) in the dataset, leading to reduced predictive performance for this group. Future work should:
- Collect additional MODERATE-risk samples
- Apply SMOTE (Synthetic Minority Over-sampling Technique)
- Implement class-weighted training
- Consider combining MODERATE with adjacent categories for binary classification

**Model Bias:** Most MODERATE misclassifications were assigned to HIGH risk, which is clinically conservative (false alarms rather than missed cases) but may lead to over-intervention.

---

## 📚 Methodology Summary for Thesis

### Data Preprocessing
- Missing value imputation (median for numeric features, mode for categorical)
- Ordinal encoding for categorical variables
- Feature scaling not required (tree-based models are scale-invariant)

### Model Training
- Algorithm: LightGBM Classifier
- Training samples: 528 (70%)
- Validation: 4-fold stratified cross-validation
- Optimization: RandomizedSearchCV with 20 iterations
- Evaluation metric: Weighted F1-score

### Model Evaluation
- Test samples: 226 (30%)
- Metrics: Accuracy, Precision, Recall, F1-score per class
- Interpretability: Feature importance (built-in) + SHAP values

### Software Stack
- **Python 3.8+**
- **Scikit-learn 1.7.2** (preprocessing, baseline models, evaluation)
- **LightGBM 4.6.0** (gradient boosting classifier)
- **Pandas 2.3.3** & **NumPy 2.3.4** (data manipulation)
- **Matplotlib 3.10.7** (visualization)
- **SHAP** (model interpretability)
- **Joblib** (model persistence)

---

## 🎯 Conclusion

This research successfully demonstrates that **machine learning algorithms can accurately predict dental caries risk in children** based on structured 7-day diet diary data. The optimized LightGBM model achieved 91.6% accuracy with exceptional performance in identifying HIGH-risk cases (F1 = 0.97), making it suitable for clinical deployment as a decision support tool in pediatric dentistry.

The model's strong reliance on the Diet Score feature validates the clinical hypothesis that dietary patterns are the primary determinant of caries risk, supporting the use of dietary interventions as a first-line preventive strategy. With further refinement to address class imbalance, this automated risk assessment system can enhance early detection and enable personalized, data-driven dental care for children.

---

## 📧 Contact

**Dr. Drushika Dinesh**  
PG Student - Dissertation Research

---

*This README documents the methodology, implementation, and results of a machine learning-based caries risk prediction system developed as part of a postgraduate research dissertation.*

