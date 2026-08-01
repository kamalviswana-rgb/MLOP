import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("tourism_project/data")
DEPLOYMENT_DIR = Path("tourism_project/deployment")

DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

TARGET = "ProdTaken"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# --------------------------------------------------
# Feature Processing
# --------------------------------------------------

categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    (StandardScaler(), numerical_cols),
)

# --------------------------------------------------
# Model
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
            ),
        ),
    ]
)

# --------------------------------------------------
# Hyperparameter Grid
# --------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
}

# --------------------------------------------------
# MLflow
# --------------------------------------------------

mlflow.set_experiment("tourism_package_prediction")

with mlflow.start_run():

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # ---------------------------
    # Log Parameters
    # ---------------------------

    mlflow.log_params(grid_search.best_params_)

    # ---------------------------
    # Log Metrics
    # ---------------------------

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # ---------------------------
    # Log Model
    # ---------------------------

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
    )

    print("Best Parameters")
    print(grid_search.best_params_)

    print("\nEvaluation Metrics")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

# --------------------------------------------------
# Save Best Model
# --------------------------------------------------

model_path = DEPLOYMENT_DIR / "tourism_model.pkl"

joblib.dump(
    best_model,
    model_path,
)

print(f"\nModel saved to: {model_path}")
