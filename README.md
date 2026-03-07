# Financial Fraud Detection

## Overview
This project is a machine learning pipeline designed to detect fraudulent financial transactions. It processes a highly imbalanced synthetic dataset and utilizes binary classification to distinguish between legitimate and fraudulent activities. 

## Dataset
The model is trained on `synthetic_fraud_dataset.csv`. Because the dataset has a severe class imbalance (only ~5% fraud cases), specific evaluation metrics and resampling techniques were necessary to ensure the model actually learns fraud patterns rather than simply predicting the majority class.

## Technologies Used
* **Python**
* **Pandas & NumPy:** Data manipulation and numerical analysis.
* **Scikit-Learn:** Building the `LogisticRegression` model and calculating evaluation metrics (Precision, Recall, F1-Score, Confusion Matrix).
* **Imbalanced-Learn:** Utilizing SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data.
* **Joblib:** Exporting and saving the trained model and preprocessor for future deployment.

## Project Structure
* `Data-preprocessing.ipynb`: Handles data cleaning, categorical encoding, and correlation analysis. It saves the preprocessing steps as `preprocessor.joblib`.
* `model_training.ipynb`: Loads the preprocessed data, balances the training set using SMOTE, trains the Logistic Regression model, and evaluates it. The final model is saved as `model.joblib`.

## Results
The model was evaluated using metrics that prioritize the minority class:
* **Precision:** 1.0000
* **Recall:** 1.0000
* **F1-Score:** 1.0000

*Note: The perfect 1.0 evaluation scores are a byproduct of the synthetic nature of the dataset. Features such as device risk scores and IP risk scores were highly correlated with the target variable, allowing the Logistic Regression model to find a flawless mathematical boundary between classes.*

## How to Run
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn joblib

<h2>Author</h2>Mithul Krishna 2nd Year, B.Tech CSE<br>NIT Bhopal
