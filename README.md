# Titanic Survival Prediction Model

A machine learning pipeline to predict Titanic passenger survival using exploratory data analysis, feature engineering, and multiple classification models.

## Overview

This project implements a complete ML workflow for binary classification: predicting whether a passenger survived the Titanic disaster based on demographic and voyage data.

- **Best CV Accuracy**: ~83.9% (Random Forest / Gradient Boosting)
- **Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVC, Soft Voting Ensemble
- **Cross-Validation**: 5-fold stratified (preserves 38% positive class ratio)

## Dataset

### Features
- **Demographics**: Sex, Age, Pclass (passenger class)
- **Family**: SibSp (siblings/spouses), Parch (parents/children)
- **Voyage**: Embarked (port), Ticket, Fare
- **Metadata**: PassengerId, Name, Cabin

### Target
- **Survived**: Binary (0 = Did not survive, 1 = Survived)
  - Class distribution: 38.4% survived, 61.6% died

### Missing Values
- Age: 263 missing (imputed by Title + Pclass median)
- Cabin: 77% missing (deck letter extracted, U for unknown)
- Embarked: 2 missing (filled with mode)
- Fare: 1 missing in test (filled with median)

## Project Structure

```
titanic-machine-learning/
├── data/
│   ├── train.csv                                    # 891 passengers, labeled
│   ├── test.csv                                     # 418 passengers, unlabeled
│   └── gender_submission.csv                        # Output schema reference
├── notebooks/
│   └── titanic-survival-prediction-model.ipynb      # Main notebook
├── outputs/
│   └── submission.csv                               # Final predictions (generated)
├── requirements.txt                                 # Python dependencies
└── README.md
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
jupyter notebook notebooks/titanic-survival-prediction-model.ipynb
```

Then in Jupyter:
- Click **Cell** → **Run All** to execute all cells
- Or press **Shift+Enter** to run cells sequentially

### Run Non-Interactively

```bash
jupyter nbconvert --to notebook --execute notebooks/titanic-survival-prediction-model.ipynb
```

### Output

The notebook generates `submission.csv` with predictions for all 418 test passengers.

## Notebook Sections

### Section 0: Imports & Data Loading
- Loads train.csv and test.csv
- Combines data for unified feature engineering (prevents train/test skew)
- Defines `style_table()` helper for consistent table formatting

### Section 1: Exploratory Data Analysis
- Survival rate analysis (38.4% positive class)
- Countplots for categorical features (Sex, Pclass, Embarked)
- Histplots for numerical features (Age, Fare)
- Heatmap: survival rate by Pclass × Sex
- Missing value summary

### Section 2: Feature Engineering
Transforms raw features into predictive signals on combined train+test data:

| Feature | Source | Method |
|---------|--------|--------|
| Title | Name | Regex extraction + explicit mapping of rare titles |
| Age (imputed) | Age | Median by Title + Pclass group |
| Deck | Cabin | Extract first letter (A-G, T, U for unknown) |
| FamilySize | SibSp + Parch | Direct sum + 1 |
| IsAlone | FamilySize | Binary flag (1 if alone) |
| LogFare | Fare | log1p transform (handles Fare=0) |
| FareBin | Fare | Quantile-based binning (4 bins) |
| AgeBin | Age | Cut into 5 age groups (Child, Teen, YoungAdult, Adult, Senior) |
| Pclass_1_Female | Pclass, Sex | Interaction: 1st class women (strongest survival signal) |
| Pclass_2_Female | Pclass, Sex | Interaction: 2nd class women |
| Pclass_3_Male | Pclass, Sex | Interaction: 3rd class men (lowest survival) |

### Section 3: Preprocessing
- Selects 17 features (raw + engineered)
- One-hot encodes categoricals (Sex, Title, Deck, FareBin, AgeBin, Embarked) with `drop_first=False`
- Results in 39 encoded features
- Aligns test set columns to training set (handles unseen categories)

### Section 4: Model Training
Trains 5 individual models + 1 ensemble with 5-fold stratified cross-validation:

| Model | Configuration | CV Accuracy |
|-------|---------------|-------------|
| Random Forest | n_est=200, depth=6, min_leaf=2 | ~83.9% |
| Gradient Boosting | n_est=200, lr=0.05, depth=4 | ~83.9% |
| XGBoost | n_est=200, lr=0.05, depth=4 | ~83.1% |
| SVC | RBF kernel, C=1.0, scaled | ~82.9% |
| Logistic Regression | StandardScaler + default C | ~82.6% |
| Soft Voting Ensemble | All 5 models, soft voting | ~83.8% |

### Section 5: Evaluation
- Boxplot comparing CV accuracy across all models
- Confusion matrix on training data (best model)
- Classification report (precision, recall, F1)
- Feature importance bar chart (top 15 features)

### Section 6: Final Prediction & Submission
- Selects best model by CV mean accuracy
- Generates predictions for all 418 test passengers
- Saves `submission.csv` and validates against expected schema

## Key Implementation Details

1. **Unified feature engineering**: Train and test combined before imputation/encoding to prevent skew
2. **Deck extraction**: 9-category encoding from Cabin (A-G, T, U) — captures survival-correlated deck location
3. **Explicit title mapping**: Rare titles mapped to Mr/Mrs/Miss/Rare instead of frequency-based lumping
4. **Stratified cross-validation**: Preserves 38/62% class balance across all 5 folds
5. **Column alignment**: `X_test.reindex(columns=X_train.columns, fill_value=0)` handles unseen categories
6. **Log transform**: `np.log1p()` for Fare (one passenger has Fare=0)
7. **Scaling**: Only applied to Logistic Regression and SVC via Pipeline (tree models don't need it)
8. **No data leakage**: Frequency-based features avoided; only imputation uses combined data
9. **Reproducibility**: All models use `random_state=42`

## Verification

After running the notebook, verify the output:

```bash
python3 -c "
import pandas as pd
s = pd.read_csv('outputs/submission.csv')
assert s.shape == (418, 2), 'Wrong shape'
assert list(s.columns) == ['PassengerId', 'Survived'], 'Wrong columns'
assert set(s['Survived']).issubset({0, 1}), 'Invalid values'
print('Valid submission')
print(f'Predicted survival rate: {s[\"Survived\"].mean():.2%}')
"
```

## Dependencies

- `pandas>=2.0` — Data manipulation
- `numpy>=1.24` — Numerical operations
- `scikit-learn>=1.3` — ML models and preprocessing
- `matplotlib>=3.7` — Static plotting
- `seaborn>=0.12` — Statistical visualizations
- `jupyter>=1.0` — Notebook environment
- `xgboost>=2.0` — Gradient boosting

## References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
