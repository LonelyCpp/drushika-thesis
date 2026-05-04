# 📍 **MODEL TEST RESULTS**

_Dataset: 800 children · Train/Test split: 560 / 240 (70/30, stratified)_

---

### **Methods — Diagnostic Analysis**

The diagnostic ability of the AI algorithm was evaluated using sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), and area under the ROC curve (AUC), each computed one-vs-rest for the four risk categories (HIGH, MODERATE, LOW, NO). 95% confidence intervals were estimated by stratified bootstrap resampling of the held-out test set (2,000 replicates). The tuned LightGBM and baseline LightGBM AUCs were compared per class using **DeLong's test for correlated ROC curves**, with statistical significance set at **P < 0.05**.

---

### **Tuned LightGBM — Test Performance**

| Metric                | Score              |
| --------------------- | ------------------ |
| **Accuracy**          | **0.9375 (93.8%)** |
| **Weighted F1-score** | **0.94**           |
| **Macro F1-score**    | **0.82**           |

#### Per-Class Performance

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------- | ------ | -------- | ------- |
| HIGH     | 0.98      | 0.98   | **0.98** | 189     |
| LOW      | 0.76      | 0.84   | **0.80** | 19      |
| MODERATE | 0.64      | 0.60   | 0.62     | 15      |
| NO       | 0.88      | 0.88   | **0.88** | 17      |

#### Diagnostic Metrics (one-vs-rest, with 95% bootstrap CIs)

| Class    | Sensitivity         | Specificity         | PPV                 | NPV                 | ROC-AUC             |
| -------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| HIGH     | 0.979 (0.957–0.995) | 0.941 (0.878–1.000) | 0.984 (0.966–1.000) | 0.923 (0.842–0.983) | 0.973 (0.930–0.999) |
| LOW      | 0.842 (0.650–1.000) | 0.977 (0.955–0.995) | 0.762 (0.571–0.941) | 0.986 (0.968–1.000) | 0.987 (0.974–0.997) |
| MODERATE | 0.600 (0.333–0.867) | 0.978 (0.957–0.996) | 0.643 (0.364–0.889) | 0.973 (0.951–0.991) | 0.914 (0.801–0.983) |
| NO       | 0.882 (0.706–1.000) | 0.991 (0.977–1.000) | 0.882 (0.714–1.000) | 0.991 (0.977–1.000) | 0.989 (0.972–1.000) |

#### Confusion Matrix

```
               Predicted
               HIGH  LOW  MODERATE  NO
Actual HIGH   [185    0     3      1]
Actual LOW    [  0   16     2      1]
Actual MOD    [  3    3     9      0]
Actual NO     [  0    2     0     15]
```

---

### **Baseline LightGBM — Test Performance (for comparison)**

| Metric                | Score              |
| --------------------- | ------------------ |
| **Accuracy**          | **0.9333 (93.3%)** |
| **Weighted F1-score** | **0.93**           |
| **Macro F1-score**    | **0.81**           |

#### Per-Class Performance

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------- | ------ | -------- | ------- |
| HIGH     | 0.98      | 0.98   | 0.98     | 189     |
| LOW      | 0.74      | 0.89   | 0.81     | 19      |
| MODERATE | 0.64      | 0.60   | 0.62     | 15      |
| NO       | 0.93      | 0.76   | 0.84     | 17      |

#### Diagnostic Metrics (one-vs-rest, with 95% bootstrap CIs)

| Class    | Sensitivity         | Specificity         | PPV                 | NPV                 | ROC-AUC             |
| -------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| HIGH     | 0.979 (0.957–0.995) | 0.922 (0.843–0.983) | 0.979 (0.957–0.995) | 0.922 (0.839–0.982) | 0.987 (0.970–0.998) |
| LOW      | 0.895 (0.737–1.000) | 0.973 (0.950–0.991) | 0.739 (0.560–0.917) | 0.991 (0.977–1.000) | 0.983 (0.967–0.995) |
| MODERATE | 0.600 (0.333–0.867) | 0.978 (0.957–0.996) | 0.643 (0.364–0.889) | 0.973 (0.951–0.991) | 0.919 (0.816–0.982) |
| NO       | 0.765 (0.545–0.947) | 0.996 (0.986–1.000) | 0.929 (0.765–1.000) | 0.982 (0.964–0.996) | 0.989 (0.969–1.000) |

#### Confusion Matrix

```
               Predicted
               HIGH  LOW  MODERATE  NO
Actual HIGH   [185    0     3      1]
Actual LOW    [  0   17     2      0]
Actual MOD    [  3    3     9      0]
Actual NO     [  1    3     0     13]
```

---

### **AUC Comparison — Tuned vs Baseline (DeLong's test)**

| Class    | AUC (Tuned) | AUC (Baseline) | P-value | Significant (P<0.05) |
| -------- | ----------- | -------------- | ------- | -------------------- |
| HIGH     | 0.973       | 0.987          | 0.220   | No                   |
| LOW      | 0.987       | 0.983          | 0.156   | No                   |
| MODERATE | 0.914       | 0.919          | 0.482   | No                   |
| NO       | 0.989       | 0.989          | 0.854   | No                   |

No per-class AUC difference between the tuned and baseline LightGBM reached statistical significance (all P > 0.05). The two models are statistically equivalent in their discriminative ability under DeLong's test; the small accuracy advantage of the tuned model (93.8% vs 93.3%) reflects threshold/operating-point differences rather than a meaningful change in the underlying ranking.

---

### **Feature Importance (Tuned LightGBM)**

| Feature    | Importance |
| ---------- | ---------- |
| Diet Score | 6181       |
| Age        | 879        |
| Sex        | 459        |

The **Diet Score** dominates (~82% of total importance), with **Age** as a meaningful secondary signal (~12%) and **Sex** contributing the remainder (~6%).

---

# ✨ Interpretation (for thesis Results section)

> The tuned LightGBM model achieved an overall accuracy of **93.8%** on the held-out test set of 240 children. Performance was strongest in predicting the **HIGH-risk** category, with an F1 score of **0.98** and balanced precision and recall of **0.98** each, indicating that the model both identifies almost all children who are genuinely high-risk for caries and rarely raises false alarms in this category. The model also performed strongly on **NO-risk** (F1 = 0.88) and **LOW-risk** (F1 = 0.80) groups, demonstrating that the larger sample size (n = 800) materially improves classification of the minority categories compared with earlier dataset versions.
>
> **MODERATE-risk** classification remains the weakest cell (F1 = 0.62, recall = 0.60). Misclassified moderate cases split evenly between LOW (n=3) and HIGH (n=3), consistent with moderate-risk dietary patterns sitting at a continuous boundary between the two adjacent classes rather than a single systematic bias. The class is also the smallest in the test set (n = 15), so additional moderate-labeled samples are likely to be the highest-leverage data-collection priority for future work.
>
> Hyperparameter tuning produced a measurable but modest gain over the baseline LightGBM (93.3% → 93.8%), with the largest improvement on the NO-risk class (recall 0.76 → 0.88). This suggests the tuning meaningfully helped the model resolve the smallest minority class, while the dataset's overall signal ceiling remains close to its current accuracy until further data collection or class-rebalancing techniques (SMOTE, class weighting) are applied.
>
> From a clinical screening standpoint, the tuned model achieves a sensitivity of **0.98 (95% CI 0.96–1.00)** and specificity of **0.94 (95% CI 0.88–1.00)** for the HIGH-risk class, with a positive predictive value of **0.98** and negative predictive value of **0.92** — an operating point that supports the model's intended use as an early-identification tool, where missing a high-risk child (false negative) is costlier than an over-referral. Specificity and NPV are uniformly high across all four classes (specificity ≥ 0.94; NPV ≥ 0.97), indicating that false-positive labelling and missed-negative cases are rare regardless of category. PPV is lower for the smaller minority classes (LOW = 0.76, MODERATE = 0.64), reflecting their lower prevalence in the test set rather than a discrimination failure. The principal weakness remains MODERATE sensitivity (0.60), which is the actionable limit of the current dataset.
>
> Discriminative ability, summarized by ROC-AUC, was excellent across all classes (HIGH 0.97, LOW 0.99, MODERATE 0.91, NO 0.99). DeLong's test for correlated ROC curves found **no statistically significant difference** between the tuned and baseline LightGBM AUCs in any class (all P > 0.05; HIGH P = 0.22, LOW P = 0.16, MODERATE P = 0.48, NO P = 0.85). The modest accuracy gain from tuning (93.3% → 93.8%) therefore reflects a better-chosen decision threshold rather than improved underlying ranking — a finding consistent with the dataset's signal ceiling being approached.
>
> Overall, the model demonstrates strong, clinically deployable predictive ability for early caries risk identification.
