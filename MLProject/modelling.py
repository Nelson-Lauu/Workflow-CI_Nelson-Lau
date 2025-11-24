import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ============================================================
# 1. Ambil argumen dari MLproject
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)

args = parser.parse_args()

# ============================================================
# 2. Load dataset preprocessing
# ============================================================
df = pd.read_csv(args.data_path)

# Target: rata-rata nilai
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

X = df.drop(columns=["avg_score"])
y = df["avg_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=args.test_size,
    random_state=args.random_state
)

# ============================================================
# 3. Start MLflow run
# ============================================================
mlflow.set_experiment("CI_Workflow_Model")

with mlflow.start_run():

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=args.random_state
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    # Logging parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("random_state", args.random_state)

    # Logging metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Save model
    mlflow.sklearn.log_model(model, "model")

print("Training selesai ✔ R2 =", r2)
