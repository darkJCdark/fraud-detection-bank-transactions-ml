# Fraud Detection in Banking Transactions using Machine Learning

This repository contains the development of a machine learning model for detecting fraudulent banking transactions.  
The project addresses the issue of digital fraud in financial institutions, proposing a predictive approach that surpasses the limitations of traditional rule-based systems.

## Contents
- **Academic documentation**: Project presentation and report including context, objectives, methodology, results, and conclusions.  
- **Source code (Python/Scikit-learn)**: Implementation of the model with preprocessing steps, correlation analysis, variable scaling, categorical encoding, and algorithm training.  
- **Evaluated models**: Logistic Regression and Random Forest, with hyperparameter optimization.  
- **Results**: The Random Forest model achieved a recall of 95%, detecting almost all fraud cases in the dataset.  
- **Ethical implications**: Considerations on data privacy, bias, transparency, and security against adversarial attacks.  

## Objective
Develop a predictive system capable of identifying patterns in banking transactions and anticipating potential fraud, reducing false negatives and strengthening trust in financial systems.

## Dataset
A public dataset (`fraud_data.csv`) was used, containing banking transaction information with temporal, geographic, demographic, and financial attributes.

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)  
- Supervised classification algorithms  
- GridSearchCV for hyperparameter optimization  
