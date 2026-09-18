# Customer Churn ML Comparison

## Project Overview

This project builds a complete **Customer Churn Machine Learning workflow** using Python and Scikit-learn.

The project compares multiple classification algorithms and includes data preprocessing, feature scaling, categorical encoding, model evaluation, cross-validation, hyperparameter tuning, RandomizedSearchCV, and feature importance analysis.

The dataset is generated programmatically with 1,000 customer records.

## Dataset Features

The dataset contains the following features:

* Age
* Tenure
* MonthlyCharges
* SupportCalls
* ContractLength
* TotalCharges
* InternetService
* PaymentMethod
* NumServices
* Churn — target variable

## Data Preprocessing

The project includes:

* Separation of features (`X`) and target (`y`)
* Train-test split
* Stratified sampling
* Numerical feature scaling using `StandardScaler`
* Categorical feature encoding using `OneHotEncoder`
* `ColumnTransformer` for combined preprocessing
* `Pipeline` for preprocessing and model training

## Machine Learning Models

The following classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. K-Nearest Neighbors (KNN)
6. Naive Bayes

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report

### Base Model Comparison

The project produced the following test-set results:

| Model               | Accuracy | F1 Score | ROC-AUC |
| ------------------- | -------: | -------: | ------: |
| Gradient Boosting   |    0.745 |    0.769 |   0.821 |
| Random Forest       |    0.730 |    0.757 |   0.815 |
| Naive Bayes         |    0.715 |    0.751 |   0.807 |
| Logistic Regression |    0.710 |    0.739 |   0.820 |
| KNN                 |    0.685 |    0.722 |   0.742 |
| Decision Tree       |    0.650 |    0.673 |   0.652 |

These results represent this particular generated dataset and test split.

## Cross-Validation

Five-fold cross-validation was implemented using `cross_val_score`.

Example Random Forest cross-validation results:

```text
[0.775   0.73125 0.76875 0.7625  0.71875]

Mean Cross-Validation Accuracy:
0.75125
```

Cross-validation was used to evaluate model performance across multiple training-validation splits rather than relying on a single split.

## Hyperparameter Tuning

### GridSearchCV

GridSearchCV was implemented for:

* Decision Tree
* Random Forest

The Decision Tree search tested parameters such as:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

The Random Forest search tested:

* `n_estimators`
* `max_depth`
* `min_samples_split`

### RandomizedSearchCV

RandomizedSearchCV was also implemented for Random Forest.

The search explored combinations of:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

Example result:

```text
Best Parameters:
n_estimators = 150
max_depth = 3
min_samples_split = 4
min_samples_leaf = 1

Best Cross-Validation F1 Score:
0.7920

Test Accuracy:
0.76
```

The cross-validation F1 score and test accuracy are different evaluation metrics and are reported separately.

## Feature Importance

Feature importance was examined for both Decision Tree and Random Forest models.

Important features identified in the trained models included:

* SupportCalls
* MonthlyCharges
* Tenure
* TotalCharges
* NumServices
* Age
* ContractLength

Feature importance was used to understand which input features contributed more strongly to the model's predictions.

## Technologies

* Python
* NumPy
* Pandas
* Scikit-learn

## Project File

```text
Customer_Churn_ML_Comparison/
│
├── customer_churn_ml.py
├── README.md
└── .gitignore
```

## How to Run

Clone the repository and run:

python customer_churn_ml.py

The program generates the dataset, performs preprocessing, trains the models, evaluates them, performs cross-validation and hyperparameter tuning, and displays the final results.

## Key Learning

This project demonstrates a complete classification workflow from data generation and preprocessing to model comparison, evaluation, cross-validation, hyperparameter tuning, and feature importance analysis.
