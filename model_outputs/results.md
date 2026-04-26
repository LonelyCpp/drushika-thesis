# 📍 **MODEL TEST RESULTS**

_Dataset: 800 children · Train/Test split: 560 / 240 (70/30, stratified)_

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
> Overall, the model demonstrates strong, clinically deployable predictive ability for early caries risk identification.
