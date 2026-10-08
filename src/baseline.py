
import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prepare_data import load_data

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Bike_Sharing_Forecasting")

# Load shared dataset
X_train, X_test, y_train, y_test, train_df, test_df = load_data()

# Baseline: predict today's rentals using yesterday's count
y_pred = X_test["yesterday_cnt"]

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MLflow tracking
with mlflow.start_run(run_name="Baseline_Yesterday"):

    mlflow.set_tag("model_type", "Baseline")

    mlflow.log_param("method", "Yesterday's rentals")
    mlflow.log_param("feature", "yesterday_cnt")
    mlflow.log_param("train_required", False)

    mlflow.log_param(
        "test_period",
        f"{test_df['dteday'].min().date()} to "
        f"{test_df['dteday'].max().date()}"
    )

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)

print(f"Baseline MAE: {mae:.2f}")
print(f"Baseline RMSE: {rmse:.2f}")
