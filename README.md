# Customer Churn ML Comparison

A machine learning project to predict customer churn using multiple classification algorithms and compare their performance.

## Project Overview

This project builds a complete customer churn prediction workflow using Python and Scikit-learn.

The project includes data preprocessing, multiple classification models, model evaluation, cross-validation, hyperparameter tuning, feature importance, handling imbalanced data, and model persistence.

## Dataset

The dataset contains 1,000 customer records with the following features:

* Age
* Tenure
* MonthlyCharges
* SupportCalls
* ContractLength
* TotalCharges
* InternetService
* PaymentMethod
* NumServices
* Churn - Target variable

The dataset was generated for this machine learning project.

## Data Preprocessing

The project uses:

* Train-test split
* StandardScaler for numerical features
* OneHotEncoder for categorical features
* ColumnTransformer
* Scikit-learn Pipelines
* Stratified train-test split

## Machine Learning Models

The following classification models were implemented and compared:

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

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Gradient Boosting   |    0.745 |     0.787 |  0.752 |    0.769 |   0.821 |
| Random Forest       |    0.730 |     0.771 |  0.743 |    0.757 |   0.815 |
| Naive Bayes         |    0.715 |     0.741 |  0.761 |    0.751 |   0.807 |
| Logistic Regression |    0.710 |     0.752 |  0.726 |    0.739 |   0.820 |
| KNN                 |    0.685 |     0.719 |  0.726 |    0.722 |   0.742 |
| Decision Tree       |    0.650 |     0.713 |  0.637 |    0.673 |   0.652 |

The base model comparison identified Gradient Boosting as the model with the highest F1 Score in this experiment.

## Cross-Validation

5-fold cross-validation was performed for Random Forest.

Mean cross-validation accuracy:

```text
0.75125
```

Individual fold scores:

```text
0.775
0.73125
0.76875
0.76250
0.71875
```

## Hyperparameter Tuning

### GridSearchCV

GridSearchCV was used for Decision Tree and Random Forest.

Decision Tree best parameters:

```text
max_depth = 3
min_samples_leaf = 1
min_samples_split = 2
```

Random Forest best parameters:

```text
max_depth = 3
min_samples_split = 2
n_estimators = 50
```

### RandomizedSearchCV

RandomizedSearchCV was also implemented for Random Forest.

Best parameters:

```text
n_estimators = 150
min_samples_split = 4
min_samples_leaf = 1
max_depth = 3
```

Best cross-validation F1 Score:

```text
0.79205
```

Test accuracy:

```text
0.76
```

## Feature Importance

Feature importance was analyzed using Decision Tree and Random Forest.

Important features included:

* SupportCalls
* MonthlyCharges
* Tenure
* TotalCharges
* NumServices
* Age
* ContractLength

SupportCalls was the most important feature in the Decision Tree feature-importance analysis.

## Handling Imbalanced Data

Two techniques were implemented:

### Class Weights

Random Forest was trained using:

```python
class_weight="balanced"
```

Results:

```text
Accuracy: 0.745
Precision: 0.816
Recall: 0.708
F1 Score: 0.758
```

### SMOTE

SMOTE was used to generate synthetic samples for the minority class.

Results:

```text
Accuracy: 0.725
Precision: 0.796
Recall: 0.690
F1 Score: 0.739
```

The results show that SMOTE did not improve the F1 Score compared with the base model in this particular experiment.

## Model Persistence

The trained SMOTE Random Forest pipeline was saved using two Python serialization methods.

### Joblib

```text
churn_model.joblib
```

The saved pipeline was successfully loaded and used for prediction.

### Pickle

```text
churn_model.pkl
```

The saved pipeline was successfully loaded and used for prediction.

Both loaded models produced the same prediction results:

```text
Accuracy: 0.725
F1 Score: 0.739
```

The complete preprocessing and model pipeline was saved so that preprocessing and prediction can be reused together.

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Imbalanced-learn
* Joblib
* Pickle

## Project Files

```text
Customer_Churn_ML_Comparison/
│
├── customer_churn_ml.py
├── README.md
├── churn_model.joblib
├── churn_model.pkl
└── .gitignore
```

## How to Run

Create and activate the virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the required libraries:

```powershell
pip install numpy pandas scikit-learn imbalanced-learn joblib
```

Run the project:

```powershell
python customer_churn_ml.py
```

## Key Learning

This project provided practical experience with:

* Data preprocessing
* Feature scaling
* Categorical encoding
* Classification algorithms
* Model evaluation
* Cross-validation
* GridSearchCV
* RandomizedSearchCV
* Feature importance
* Class weights
* SMOTE
* Model persistence using Joblib
* Model persistence using Pickle
* Comparing multiple machine learning models
