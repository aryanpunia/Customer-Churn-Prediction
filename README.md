# Customer Churn Prediction Project

## Overview

This project aims to predict whether a customer will leave the bank (churn) or stay, based on various customer attributes. Accurate predictions can help banks take proactive measures to retain valuable customers.

## Dataset

The dataset includes the following features:

- **Customer ID**: Unique identifier for each customer
- **Surname**: Customer's surname
- **CreditScore** : Customer's credit score
- **Geography**: Customer's country
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Tenure**: Number of years the customer has been with the bank
- **Balance**: Account balance
- **NumOfProducts**: Number of bank products the customer is using
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No)
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No)
- **EstimatedSalary**: Customer's estimated salary
- **Exited**: Whether the customer left the bank (1 = Yes, 0 = No)

***Usage***
**Data Preprocessing**:

Load the dataset and preprocess it by handling missing values, encoding categorical variables, and scaling numerical features.

**Model Training**:

Train various machine learning models (e.g., Logistic Regression, Decision Trees, Random Forest, etc.) on the processed data.

**Model Evaluation**:

Evaluate the models using appropriate metrics (e.g., accuracy, precision, recall, F1 score) and select the best-performing model.

**Prediction**:

Use the trained model to predict whether new customers will churn or stay.

**Results**
The project achieved an accuracy of **85%** on the test dataset.
The best-performing model was Random Forest classifier, with the following metrics:
Accuracy: 85%

