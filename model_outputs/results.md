# 📍 **MODEL TEST RESULTS RECEIVED**

### **Tuned LightGBM Test Performance**

| Metric                | Score              |
| --------------------- | ------------------ |
| **Accuracy**          | **0.9163 (91.6%)** |
| **Weighted F1-score** | **0.90**           |

---

### **Per-Class Performance**

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------- | ------ | -------- | ------- |
| HIGH     | 0.95      | 0.98   | 0.97     | 177     |
| LOW      | 0.78      | 0.90   | 0.84     | 20      |
| MODERATE | 0.50      | 0.14   | 0.22     | 14      |
| NO       | 0.82      | 0.88   | 0.85     | 16      |

### **Confusion Matrix**

```
               Predicted
               H   L   M   N
Actual HIGH   [174  0   1   2]
Actual LOW    [ 0  18   1   1]
Actual MOD    [ 9   3   2   0]
Actual NO     [ 0   2   0  14]
```

---

# ✨ Interpretation (to include in your thesis Results section)

> The tuned LightGBM model achieved an overall accuracy of **91.6%** on the held-out test dataset. Performance was strongest in predicting the **HIGH-risk** category, with an F1 score of **0.97** and recall of **0.98**, indicating the model successfully identifies almost all children who are actually high-risk for caries. The model also performed well on **LOW** and **NO** risk groups with F1 scores of **0.84** and **0.85**, respectively.
> Prediction performance for the **MODERATE** class was lower (F1 = 0.22), likely due to limited representation in the dataset (**n = 14**) causing class imbalance.
> The **confusion matrix** showed that most misclassifications for moderate samples were incorrectly assigned to the **HIGH** class, suggesting similarity in feature patterns.
> Overall, the model demonstrates strong predictive ability for early caries risk identification and is suitable for clinical/educational deployment after addressing class imbalance (e.g. SMOTE, class weighting, or collecting more moderate-labeled samples).