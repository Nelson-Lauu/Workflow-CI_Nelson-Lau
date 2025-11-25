import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import time
import os

# ============================================================
# 1. Ambil argumen dari MLProject
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# ============================================================
# 2. Load Data
# ============================================================
df = pd.read_csv(args.data_path)

# Target = rata-rata 3 nilai
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

X = df.drop(columns=["avg_score"])
y = df["avg_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# ============================================================
# 3. MLflow config — FIX UTAMA
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKING_DIR = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri("file://" + TRACKING_DIR)
mlflow.set_experiment("CI_Workflow_Model")

with mlflow.start_run():

    # ================= Model =================
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # ================= Training =================
    start = time.time()
    model.fit(X_train, y_train)
    inference_time = time.time() - start

    # ================= Predict =================
    preds = model.predict(X_test)

    # ================= Metrics =================
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    # ================= Logging =================
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("random_state", args.random_state)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Inference_Time", inference_time)

    # ================= Simpan Model =================
    mlflow.sklearn.log_model(model, artifact_path="model")

print("Training CI selesai. R2 =", r2)
