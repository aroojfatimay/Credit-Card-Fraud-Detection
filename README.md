# Credit Card Fraud Detection

## Project Overview

This project detects fraudulent credit card transactions using supervised machine learning techniques.

The project focuses on handling highly imbalanced data and building a reliable fraud detection pipeline using SMOTE, Logistic Regression, and Random Forest.

## Dataset

The project uses the `creditcard.csv` dataset containing **284,807 credit card transactions**.

The `Class` column indicates:

* `0` — Legitimate transaction
* `1` — Fraudulent transaction

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Jupyter Notebook

## Project Steps

1. Load and explore the dataset.
2. Analyze the class distribution.
3. Split the data using stratified train-test split.
4. Handle class imbalance using SMOTE.
5. Train Logistic Regression and Random Forest models.
6. Tune the Random Forest model using GridSearchCV.
7. Evaluate model performance.
8. Generate confusion matrices and ROC curves.
9. Analyze Random Forest feature importance.

## Machine Learning Models

* Logistic Regression
* Random Forest Classifier

## Evaluation Metrics

The models are evaluated using:

* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

## How to Run

### 1. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 2. Open the Jupyter Notebook

Open:

`notebooks/Project_2_Fraud_Detection.ipynb`

### 3. Run the notebook

Run the notebook cells from top to bottom.

## Objective

The main objective of this project is to build a machine learning pipeline that can identify fraudulent credit card transactions while properly handling the severe class imbalance in the dataset.
