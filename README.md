# Titanic Survival Prediction Model

A comprehensive machine learning pipeline to predict Titanic passenger survival using exploratory data analysis, feature engineering, and multiple classification models.

## Overview

This project implements a complete ML workflow for binary classification: predicting whether a passenger survived the Titanic disaster based on demographic and voyage data. The pipeline trains and compares multiple models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) using stratified cross-validation.

## Dataset

### Features
- **Demographics**: Sex, Age, Pclass (passenger class)
- **Family**: SibSp (siblings/spouses), Parch (parents/children)
- **Voyage**: Embarked (port), Ticket, Fare
- **Metadata**: PassengerId, Name, Cabin

### Target
- **Survived**: Binary (0 = Did not survive, 1 = Survived)
  - Class distribution: 38.4% survived (moderately imbalanced)

### Missing Values
- Age: 177 missing (imputed by Title + Pclass median)
- Cabin: 77% missing (extracted deck letter)
- Embarked: 2 missing (filled with mode)
- Fare: 1 missing in test (filled with median)

## Project Structure

```
titanic-machine-learning/
├── data/
│   ├── train.csv              # 891 passengers, labeled
│   ├── test.csv               # 418 passengers, unlabeled
│   └── gender_submission.csv  # Output schema reference
├── notebooks/
│   └── titanic_analysis.ipynb # Main analysis notebook
├── outputs/
│   └── submission.csv         # Final predictions (generated)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd titanic-machine-learning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Analysis

```bash
jupyter notebook notebooks/titanic_analysis.ipynb
```

Then in Jupyter:
- Click **Cell** → **Run All** to execute all cells
- Or press **Shift+Enter** to run cells sequentially

### Run Non-Interactively

```bash
jupyter nbconvert --to notebook --execute notebooks/titanic_analysis.ipynb
```

### Output

The notebook generates `outputs/submission.csv` with predictions for all 418 test passengers.

## Notebook Sections

### Section 0: Imports & Data Loading
- Loads train.csv and test.csv
- Combines data for unified feature engineering (prevents train/test skew)
- Displays data shapes and types

### Section 1: Exploratory Data Analysis
- Survival rate analysis (38.4% positive class)
- Countplots for categorical features (Sex, Pclass, Embarked)
- Histplots for numerical features (Age, Fare)
- Heatmap: survival rate by Pclass × Sex
- Missing value summary

### Section 2: Feature Engineering
Transforms raw features into predictive signals:

| Feature | Source | Method |
|---------|--------|--------|
| Title | Name | Regex extraction + consolidate rare titles |
| Age | Age | Impute by Title + Pclass median |
| FamilySize | SibSp + Parch | Direct sum + 1 |
| IsAlone | FamilySize | Binary flag (1 if alone) |
| Deck | Cabin | Extract first letter (U for missing) |
| LogFare | Fare | log1p transform (handles Fare=0) |
| FareBin | Fare | Quantile-based binning (4 bins) |
| AgeBin | Age | Cut into 5 age groups |

### Section 3: Preprocessing
- Drops raw features (PassengerId, Name, Ticket, Cabin)
- One-hot encodes categorical variables
- Aligns test set columns to training set (handles unseen categories)
- Scales features for Logistic Regression only (via Pipeline)

### Section 4: Model Training
Trains 4 models with 5-fold stratified cross-validation:

| Model | Configuration |
|-------|---------------|
| Logistic Regression | max_iter=1000, scaled via Pipeline |
| Random Forest | n_estimators=200, max_depth=6, min_samples_leaf=2 |
| Gradient Boosting | n_estimators=200, learning_rate=0.05, max_depth=4 |
| XGBoost | n_estimators=200, learning_rate=0.05, max_depth=4 |

**Expected Performance**: CV accuracy ≥ 80% for best model

### Section 5: Evaluation
- Boxplot comparing CV accuracy across models
- Confusion matrix on training data
- Classification report (precision, recall, F1)
- Feature importance bar chart (top 15 features)

### Section 6: Final Prediction & Submission
- Generates predictions for test set
- Creates `outputs/submission.csv` with schema:
  ```
  PassengerId, Survived
  892, 0
  893, 1
  ...
  1309, 0
  ```

## Verification

After running the notebook, verify the output:

```bash
python3 << 'EOF'
import pandas as pd

submission = pd.read_csv('outputs/submission.csv')

# Check validity
assert submission.shape == (418, 2), "Wrong shape"
assert list(submission.columns) == ['PassengerId', 'Survived'], "Wrong columns"
assert set(submission['Survived']).issubset({0, 1}), "Invalid values"
assert submission.isnull().sum().sum() == 0, "Missing values"

print("✅ Submission valid!")
print(f"Predicted survival rate: {submission['Survived'].mean():.2%}")
EOF
```

## Key Implementation Details

1. **Unified feature engineering**: Train and test combined before imputation to prevent data leakage
2. **Stratified cross-validation**: Preserves class balance across folds
3. **Column alignment**: X_test reindexed to X_train columns to handle unseen categories
4. **Log transform**: Uses `np.log1p()` for Fare (one passenger has Fare=0)
5. **Scaling**: Only applied to Logistic Regression via Pipeline (tree models ignore it)
6. **Categorical encoding**: One-hot encoding with `pd.get_dummies()`

## Dependencies

- `pandas>=2.0` — Data manipulation
- `numpy>=1.24` — Numerical operations
- `scikit-learn>=1.3` — Machine learning models and preprocessing
- `matplotlib>=3.7` — Static plotting
- `seaborn>=0.12` — Statistical visualizations
- `jupyter>=1.0` — Notebook environment
- `xgboost>=2.0` — Gradient boosting

## Expected Output

```
outputs/submission.csv
├─ Shape: 418 rows × 2 columns
├─ Columns: PassengerId, Survived
├─ Values: 0 (did not survive) or 1 (survived)
└─ No missing values
```

## Notes

- **Imbalanced class**: 62% negative, 38% positive — consider AUC/F1 alongside accuracy
- **Reproducibility**: All models use `random_state=42` for consistent results
- **Hyperparameters**: Tuned based on cross-validation performance, not grid search

## References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**Last Updated**: March 2026
