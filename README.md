# Task 16 — Model Validation & K-Fold

AI/ML Developer · PlaceMux · Phase 1 Industry Immersion

## Objective
Validate models rigorously using cross-validation and K-Fold to ensure performance generalises, rather than trusting a single lucky (or unlucky) train/test split.

## Approach
- Used **Stratified 5-Fold Cross-Validation** to compare three candidate models on the exact same folds, ensuring a fair comparison.
- Reported **mean accuracy and standard deviation** for each model, not just the best fold.
- Used **nested cross-validation** to tune Random Forest's `n_estimators`, keeping the tuning and evaluation folds separate to avoid data leakage.

## Tools
- scikit-learn (`KFold`, `StratifiedKFold`, `cross_val_score`, `GridSearchCV`)
- numpy

## Results

| Model | Mean Accuracy | Std Dev |
|---|---|---|
| Logistic Regression | 0.9737 | 0.0166 |
| Random Forest | 0.9561 | 0.0123 |
| **SVM** | **0.9772** | 0.0163 |

**Best generalising model: SVM**

Random Forest had the lowest variance (most consistent across folds) but a lower mean accuracy, so it wasn't selected as the top performer despite its stability.

## Nested CV (Random Forest tuning)
An inner loop selected the best `n_estimators` value on training folds only; an outer loop then evaluated that tuned model on folds it never saw during tuning. This avoids the pitfall of tuning and evaluating on the same data.

## How to run

```bash
pip install scikit-learn numpy
python kfold_validation.py
```

## Dataset
Uses scikit-learn's built-in `load_breast_cancer` dataset (real, small, classification task) for demonstration.An inner loop selected the best `n_estimators` value on training folds only; an outer loop then evaluated that tuned model on folds it never saw during tuning. This avoids the pitfall of tuning and evaluating on the same data.

