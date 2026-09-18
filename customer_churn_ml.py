import numpy as np
import pandas as pd
import joblib
import pickle

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
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
# STEP 13: CROSS-VALIDATION
# ==========================================================

print("\n==================================================")
print("CROSS-VALIDATION")
print("==================================================")


rf_cv_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])


cv_scores = cross_val_score(
    rf_cv_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


print("Cross-Validation Scores:")
print(cv_scores)

print("\nMean Cross-Validation Accuracy:")
print(cv_scores.mean())


# ==========================================================
# STEP 14: RANDOMIZED SEARCH CV
# ==========================================================

print("\n==================================================")
print("RANDOMIZED SEARCH CV - RANDOM FOREST")
print("==================================================")


rf_random_params = {
    "model__n_estimators": [50, 75, 100, 125, 150, 200],
    "model__max_depth": [3, 5, 7, 10, None],
    "model__min_samples_split": [2, 4, 6, 8, 10],
    "model__min_samples_leaf": [1, 2, 3, 4]
}


rf_random = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=rf_random_params,
    n_iter=10,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)


rf_random.fit(X_train, y_train)

print("\nBest Parameters:")
print(rf_random.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(rf_random.best_score_)

best_rf_random = rf_random.best_estimator_

y_pred_best_rf_random = best_rf_random.predict(X_test)

random_test_accuracy = accuracy_score(
    y_test,
    y_pred_best_rf_random
)

print("\nRandomizedSearchCV Test Accuracy:")
print(random_test_accuracy)


# ==========================================================
# STEP 15: FINAL MODEL COMPARISON
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
# STEP 16: BEST MODEL
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
# STEP 17: DECISION TREE FEATURE IMPORTANCE
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
# STEP 18: RANDOM FOREST FEATURE IMPORTANCE
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
# STEP 19: HANDLING IMBALANCED DATA - CLASS WEIGHTS
# ==========================================================

print("\n==================================================")
print("CLASS WEIGHT - RANDOM FOREST")
print("==================================================")


rf_balanced_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        )
    )
])


rf_balanced_pipeline.fit(
    X_train,
    y_train
)


y_pred_balanced = rf_balanced_pipeline.predict(
    X_test
)


print("\nClassification Report - Balanced Random Forest:")

print(
    classification_report(
        y_test,
        y_pred_balanced,
        zero_division=0
    )
)


print("\nConfusion Matrix - Balanced Random Forest:")

print(
    confusion_matrix(
        y_test,
        y_pred_balanced
    )
)


balanced_accuracy = accuracy_score(
    y_test,
    y_pred_balanced
)

balanced_precision = precision_score(
    y_test,
    y_pred_balanced,
    zero_division=0
)

balanced_recall = recall_score(
    y_test,
    y_pred_balanced,
    zero_division=0
)

balanced_f1 = f1_score(
    y_test,
    y_pred_balanced,
    zero_division=0
)


print("\nBalanced Random Forest Metrics:")
print("Accuracy:", balanced_accuracy)
print("Precision:", balanced_precision)
print("Recall:", balanced_recall)
print("F1 Score:", balanced_f1)


# ==========================================================
# STEP 20: HANDLING IMBALANCED DATA - SMOTE
# ==========================================================

print("\n==================================================")
print("SMOTE - RANDOM FOREST")
print("==================================================")


rf_smote_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])


rf_smote_pipeline.fit(
    X_train,
    y_train
)


y_pred_smote = rf_smote_pipeline.predict(
    X_test
)


print("\nClassification Report - SMOTE Random Forest:")

print(
    classification_report(
        y_test,
        y_pred_smote,
        zero_division=0
    )
)


print("\nConfusion Matrix - SMOTE Random Forest:")

print(
    confusion_matrix(
        y_test,
        y_pred_smote
    )
)


smote_accuracy = accuracy_score(
    y_test,
    y_pred_smote
)

smote_precision = precision_score(
    y_test,
    y_pred_smote,
    zero_division=0
)

smote_recall = recall_score(
    y_test,
    y_pred_smote,
    zero_division=0
)

smote_f1 = f1_score(
    y_test,
    y_pred_smote,
    zero_division=0
)


print("\nSMOTE Random Forest Metrics:")
print("Accuracy:", smote_accuracy)
print("Precision:", smote_precision)
print("Recall:", smote_recall)
print("F1 Score:", smote_f1)


# ==========================================================
# STEP 21: MODEL PERSISTENCE - JOBLIB
# ==========================================================

print("\n==================================================")
print("MODEL PERSISTENCE - JOBLIB")
print("==================================================")


joblib.dump(
    rf_smote_pipeline,
    "churn_model.joblib"
)


print("\nModel saved successfully as churn_model.joblib")


# ==========================================================
# LOAD JOBLIB MODEL
# ==========================================================

loaded_model = joblib.load(
    "churn_model.joblib"
)


print("Model loaded successfully!")


# ==========================================================
# PREDICTION USING LOADED JOBLIB MODEL
# ==========================================================

loaded_predictions = loaded_model.predict(
    X_test
)


print("\nPredictions from loaded model:")
print(loaded_predictions[:10])


# ==========================================================
# VERIFY LOADED JOBLIB MODEL
# ==========================================================

loaded_accuracy = accuracy_score(
    y_test,
    loaded_predictions
)

loaded_f1 = f1_score(
    y_test,
    loaded_predictions,
    zero_division=0
)


print("\nLoaded Model Accuracy:")
print(loaded_accuracy)

print("\nLoaded Model F1 Score:")
print(loaded_f1)


# ==========================================================
# STEP 22: MODEL PERSISTENCE - PICKLE
# ==========================================================

print("\n==================================================")
print("MODEL PERSISTENCE - PICKLE")
print("==================================================")


# Save the model
with open("churn_model.pkl", "wb") as file:
    pickle.dump(
        rf_smote_pipeline,
        file
    )


print("\nModel saved successfully as churn_model.pkl")


# ==========================================================
# LOAD PICKLE MODEL
# ==========================================================

with open("churn_model.pkl", "rb") as file:
    pickle_model = pickle.load(file)


print("Pickle model loaded successfully!")


# ==========================================================
# PREDICTION USING PICKLE MODEL
# ==========================================================

pickle_predictions = pickle_model.predict(
    X_test
)


print("\nPredictions from Pickle model:")
print(pickle_predictions[:10])


# ==========================================================
# VERIFY PICKLE MODEL
# ==========================================================

pickle_accuracy = accuracy_score(
    y_test,
    pickle_predictions
)

pickle_f1 = f1_score(
    y_test,
    pickle_predictions,
    zero_division=0
)


print("\nPickle Model Accuracy:")
print(pickle_accuracy)

print("\nPickle Model F1 Score:")
print(pickle_f1)


# ==========================================================
# PROJECT COMPLETE
# ==========================================================

print("\n==================================================")
print("CUSTOMER CHURN ML PROJECT COMPLETE")
print("==================================================")

print("\nCompleted Topics:")
print("1. Data Preprocessing")
print("2. Classification Models")
print("3. Model Evaluation")
print("4. Cross-Validation")
print("5. GridSearchCV")
print("6. RandomizedSearchCV")
print("7. Feature Importance")
print("8. Class Weights")
print("9. SMOTE")
print("10. Joblib Model Persistence")
print("11. Pickle Model Persistence")