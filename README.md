# Financial Fraud Detection with Explainable AI (XAI)

## Overview
This project is an end-to-end machine learning pipeline designed to detect fraudulent financial transactions. It processes a highly imbalanced synthetic dataset, utilizes binary classification to distinguish between legitimate and fraudulent activities, and integrates Explainable AI (XAI) to provide transparent, interpretable model decisions.

## Dataset
The model is trained on `synthetic_fraud_dataset.csv`. Because the dataset has a severe class imbalance (~5% fraud cases), specific evaluation metrics and resampling techniques (SMOTE) were implemented to ensure the model learns actual fraud patterns rather than simply predicting the majority class.

## Technologies Used
* **Python**
* **Pandas & NumPy:** Data manipulation and numerical analysis.
* **Scikit-Learn:** Building the `LogisticRegression` model and calculating evaluation metrics (Precision, Recall, F1-Score, Confusion Matrix).
* **Imbalanced-Learn:** Utilizing SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data without data leakage.
* **SHAP & Matplotlib:** Implementing Explainable AI to visualize global feature importance and local transaction-level predictions.
* **Joblib:** Exporting and saving the trained model and preprocessor for future deployment.

## Project Structure
* `Data-preprocessing.ipynb`: Handles data cleaning, categorical encoding, and correlation analysis. Saves the preprocessing steps as `preprocessor.joblib`.
* `model_training.ipynb`: Loads the preprocessed data, balances the training set using SMOTE, trains the Logistic Regression model, and evaluates it. The final model is saved as `model.joblib`.
* `XAI.ipynb`: Loads the trained model and testing data to generate SHAP visualizations, explaining the specific features driving the model's predictions.

## Model Interpretability (XAI)
To ensure the model is not a "black box," SHAP (SHapley Additive exPlanations) was integrated:
* **Global Explainability:** Summary plots highlight the overall drivers of the model, revealing that `device_risk_score` and `ip_risk_score` are the strongest indicators of fraud.
* **Local Explainability:** Waterfall plots breakdown individual transactions, explaining to stakeholders exactly *why* a specific transaction was flagged or approved.

## Results
The model was evaluated using metrics that prioritize the minority class:
* **Precision:** 1.0000
* **Recall:** 1.0000
* **F1-Score:** 1.0000

*Note: The perfect 1.0 evaluation scores are a byproduct of the synthetic nature of the dataset. Features such as device risk scores were highly correlated with the target variable, allowing the model to find a flawless mathematical boundary.*

## How to Run
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn shap matplotlib joblib

<h2>Author</h2>Mithul Krishna 2nd Year, B.Tech CSE<br>NIT Bhopal
