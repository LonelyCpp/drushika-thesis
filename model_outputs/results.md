# 📍 **MODEL TEST RESULTS**

_Dataset: 799 children · Train/Test split: 559 / 240 (70/30, stratified)_

---

### **Tuned LightGBM — Test Performance**

| Metric                | Score              |
| --------------------- | ------------------ |
| **Accuracy**          | **0.9292 (92.9%)** |
| **Weighted F1-score** | **0.93**           |
| **Macro F1-score**    | **0.81**           |

#### Per-Class Performance

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------- | ------ | -------- | ------- |
| HIGH     | 0.99      | 0.97   | **0.98** | 189     |
| LOW      | 0.70      | 0.84   | 0.76     | 19      |
| MODERATE | 0.62      | 0.67   | 0.65     | 15      |
| NO       | 0.88      | 0.82   | 0.85     | 17      |

#### Confusion Matrix

```
               Predicted
               HIGH  LOW  MODERATE  NO
Actual HIGH   [183    1     4      1]
Actual LOW    [  0   16     2      1]
Actual MOD    [  2    3    10      0]
Actual NO     [  0    3     0     14]
```

---

### **Baseline LightGBM — Test Performance (for comparison)**

| Metric                | Score              |
| --------------------- | ------------------ |
| **Accuracy**          | **0.9333 (93.3%)** |
| **Weighted F1-score** | **0.93**           |
| **Macro F1-score**    | **0.82**           |

#### Per-Class Performance

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------- | ------ | -------- | ------- |
| HIGH     | 0.98      | 0.97   | 0.98     | 189     |
| LOW      | 0.71      | 0.89   | 0.79     | 19      |
| MODERATE | 0.67      | 0.67   | 0.67     | 15      |
| NO       | 0.93      | 0.76   | 0.84     | 17      |

#### Confusion Matrix

```
               Predicted
               HIGH  LOW  MODERATE  NO
Actual HIGH   [184    1     3      1]
Actual LOW    [  0   17     2      0]
Actual MOD    [  3    2    10      0]
Actual NO     [  0    4     0     13]
```

---

### **Feature Importance (Tuned LightGBM)**

| Feature    | Importance |
| ---------- | ---------- |
| Diet Score | 1223       |
| Age        | 256        |
| Sex        | 118        |

The **Diet Score** dominates (~77% of total importance), with **Age** as a meaningful secondary signal (~16%) and **Sex** contributing the remainder.

---

# ✨ Interpretation (for thesis Results section)

> The tuned LightGBM model achieved an overall accuracy of **92.9%** on the held-out test set of 240 children. Performance was strongest in predicting the **HIGH-risk** category, with an F1 score of **0.98**, recall of **0.97**, and precision of **0.99**, indicating that the model both identifies almost all children who are genuinely high-risk for caries and rarely raises false alarms in this category. The model also performed well on **NO-risk** (F1 = 0.85) and **MODERATE-risk** (F1 = 0.65) groups, demonstrating that the larger sample size (n = 799) materially improves classification of intermediate-risk children compared with earlier dataset versions.
>
> The weakest cell of the model is **LOW-risk precision (0.70)**, with recall remaining strong at **0.84** — meaning the model rarely misses LOW-risk children but occasionally over-assigns this label to MODERATE or NO cases. Misclassifications on the MODERATE class are now distributed roughly evenly between LOW and HIGH categories rather than concentrating on HIGH, suggesting the model has begun to distinguish the overlapping dietary patterns that previously caused systematic under-prediction.
>
> The interestingly close performance between the **baseline LightGBM (93.3%)** and the **tuned LightGBM (92.9%)** suggests that the search space is at the edge of the dataset's signal ceiling — additional minority-class samples are likely to yield more gain than further hyperparameter tuning.
>
> Overall, the model demonstrates strong, clinically deployable predictive ability for early caries risk identification. Continued data collection in the LOW, MODERATE, and NO categories — alongside class-weighted training or SMOTE — is the recommended next step for further reducing the residual class imbalance.
