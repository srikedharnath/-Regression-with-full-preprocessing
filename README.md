# -Regression-with-full-preprocessing
# Insurance Charges Prediction using SVR

## Overview

This project predicts medical insurance charges using Support Vector Regression (SVR).

The prediction is based on:
- Age
- Gender
- BMI
- Number of Children
- Smoking Status
- Region

The model is trained using machine learning techniques and evaluates prediction performance using regression metrics.

---

# Dataset Information

The dataset contains 1338 rows and 7 columns.

## Features

| Column | Description |
|---|---|
| age | Age of the person |
| sex | Gender of the person |
| bmi | Body Mass Index |
| children | Number of children |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance charges |

---

# Machine Learning Algorithm

- Support Vector Regression (SVR)

---

# Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# Project Workflow

1. Load Dataset
2. Understand Dataset
3. Check Null Values
4. Encode Categorical Data
5. Feature Scaling
6. Train-Test Split
7. Train SVR Model
8. Prediction
9. Model Evaluation

---

# Data Preprocessing

## Categorical Encoding

The following columns were encoded:
- sex
- smoker
- region

using Label Encoding.

---

# Feature Scaling

StandardScaler was used because SVR is sensitive to feature ranges.

---

# SVR Model

```python
SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
)
