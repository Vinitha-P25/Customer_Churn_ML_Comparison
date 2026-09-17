import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# CUSTOMER CHURN - MACHINE LEARNING MODEL COMPARISON
# ==========================================================


# ==========================================================
# STEP 1: CREATE LARGE DATASET
# ==========================================================

np.random.seed(42)

n = 1000

age = np.random.randint(18, 70, n)

tenure = np.random.randint(1, 61, n)

monthly_charges = np.round(
    np.random.uniform(25, 120, n), 2
)

support_calls = np.random.randint(0, 11, n)

contract_length = np.random.choice(
    [1, 12, 24],
    n,
    p=[0.45, 0.30, 0.25]
)

total_charges = np.round(
    monthly_charges * tenure +
    np.random.normal(0, 100, n),
    2
)

total_charges = np.maximum(total_charges, 0)

internet_service = np.random.choice(
    ["DSL", "Fiber", "None"],
    n,
    p=[0.35, 0.50, 0.15]
)

payment_method = np.random.choice(
    ["Electronic", "Bank Transfer", "Credit Card", "Mailed Check"],
    n,
    p=[0.40, 0.25, 0.25, 0.10]
)

num_services = np.random.randint(1, 7, n)


# ==========================================================
# STEP 2: CREATE CHURN TARGET
# ==========================================================

churn_score = (
    0.025 * (monthly_charges - 70)
    + 0.35 * support_calls
    - 0.035 * tenure
    - 0.50 * (contract_length == 24)
    - 0.20 * (contract_length == 12)
    + 0.40 * (internet_service == "Fiber")
    + 0.25 * (payment_method == "Electronic")
    - 0.10 * num_services
)

probability = 1 / (1 + np.exp(-churn_score))

churn = np.random.binomial(1, probability)


# ==========================================================
# STEP 3: CREATE DATAFRAME
# ==========================================================

df = pd.DataFrame({
    "Age": age,
    "Tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "SupportCalls": support_calls,
    "ContractLength": contract_length,
    "TotalCharges": total_charges,
    "InternetService": internet_service,
    "PaymentMethod": payment_method,
    "NumServices": num_services,
    "Churn": churn
})


print("\n==================================================")
print("CUSTOMER CHURN DATASET")
print("==================================================")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)


# ==========================================================
# STEP 4: FEATURES AND TARGET
# ==========================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]


# ==========================================================
# STEP 5: TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ==========================================================
# STEP 6: FEATURE TYPES
# ==========================================================

numerical_features = [
    "Age",
    "Tenure",
    "MonthlyCharges",
    "SupportCalls",
    "ContractLength",
    "TotalCharges",
    "NumServices"
]

categorical_features = [
    "InternetService",
    "PaymentMethod"
]


# ==========================================================
# STEP 7: PREPROCESSING
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ==========================================================
# STEP 8: DEFINE MODELS
# ==========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}


# ==========================================================
# STEP 9: TRAIN AND EVALUATE MODELS
# ==========================================================

results = []

trained_models = {}


for model_name, model in models.items():

    print("\n==================================================")
    print(model_name.upper())
    print("==================================================")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = pipeline


# ==========================================================
# STEP 10: NAIVE BAYES
# ==========================================================

print("\n==================================================")
print("NAIVE BAYES")
print("==================================================")


nb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GaussianNB())
])

nb_pipeline.fit(X_train, y_train)

y_pred_nb = nb_pipeline.predict(X_test)

y_probability_nb = nb_pipeline.predict_proba(X_test)[:, 1]

nb_accuracy = accuracy_score(y_test, y_pred_nb)

nb_precision = precision_score(
    y_test,
    y_pred_nb,
    zero_division=0
)

nb_recall = recall_score(
    y_test,
    y_pred_nb,
    zero_division=0
)

nb_f1 = f1_score(
    y_test,
    y_pred_nb,
    zero_division=0
)

nb_roc_auc = roc_auc_score(
    y_test,
    y_probability_nb
)

print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)
print("ROC-AUC:", nb_roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_nb,
        zero_division=0
    )
)

results.append({
    "Model": "Naive Bayes",
    "Accuracy": nb_accuracy,
    "Precision": nb_precision,
    "Recall": nb_recall,
    "F1 Score": nb_f1,
    "ROC-AUC": nb_roc_auc
})

trained_models["Naive Bayes"] = nb_pipeline


# ==========================================================
# STEP 11: DECISION TREE HYPERPARAMETER TUNING
# ==========================================================

print("\n==================================================")
print("DECISION TREE HYPERPARAMETER TUNING")
print("==================================================")


dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        DecisionTreeClassifier(
            random_state=42
        )
    )
])


dt_params = {
    "model__max_depth": [2, 3, 4, 5, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}


dt_grid = GridSearchCV(
    dt_pipeline,
    dt_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)

print("Best Parameters:")
print(dt_grid.best_params_)

print("\nBest Cross-Validation F1:")
print(dt_grid.best_score_)

best_dt = dt_grid.best_estimator_

y_pred_best_dt = best_dt.predict(X_test)

print("\nBest Decision Tree Test Accuracy:")
print(
    accuracy_score(
        y_test,
        y_pred_best_dt
    )
)


# ==========================================================
# STEP 12: RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================================

print("\n==================================================")
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("==================================================")


rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            random_state=42
        )
    )
])


rf_params = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [3, 5, None],
    "model__min_samples_split": [2, 5]
}


rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("Best Parameters:")
print(rf_grid.best_params_)

print("\nBest Cross-Validation F1:")
print(rf_grid.best_score_)

best_rf = rf_grid.best_estimator_

y_pred_best_rf = best_rf.predict(X_test)

print("\nBest Random Forest Test Accuracy:")
print(
    accuracy_score(
        y_test,
        y_pred_best_rf
    )
)


# ==========================================================
# STEP 13: FINAL MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n==================================================")
print("FINAL MODEL COMPARISON")
print("==================================================")

print(
    results_df.to_string(index=False)
)


# ==========================================================
# STEP 14: BEST MODEL
# ==========================================================

best_model_name = results_df.iloc[0]["Model"]

print("\n==================================================")
print("BEST MODEL")
print("==================================================")

print(
    "Best model based on F1 Score:",
    best_model_name
)


# ==========================================================
# STEP 15: DECISION TREE FEATURE IMPORTANCE
# ==========================================================

print("\n==================================================")
print("DECISION TREE FEATURE IMPORTANCE")
print("==================================================")


dt_model = best_dt.named_steps["model"]

feature_names = (
    best_dt
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": dt_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(15))


# ==========================================================
# STEP 16: RANDOM FOREST FEATURE IMPORTANCE
# ==========================================================

print("\n==================================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("==================================================")


rf_model = best_rf.named_steps["model"]

rf_feature_names = (
    best_rf
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

rf_importance_df = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance": rf_model.feature_importances_
})

rf_importance_df = rf_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(rf_importance_df.head(15))


# ==========================================================
# PROJECT COMPLETE
# ==========================================================

print("\n==================================================")
print("CUSTOMER CHURN ML PROJECT COMPLETE")
print("==================================================")