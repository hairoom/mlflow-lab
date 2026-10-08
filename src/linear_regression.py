
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prepare_data import load_data, FEATURES

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Bike_Sharing_Forecasting")

# Load data
X_train, X_test, y_train, y_test, train_df, test_df = load_data()

# Model
model = LinearRegression()

with mlflow.start_run(run_name="Linear_Regression"):

    mlflow.set_tag("model_type", "Linear Regression")
    mlflow.set_tag("analyst", "Analyst A")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Log parameters
    mlflow.log_params(model.get_params())
    mlflow.log_param("features", ", ".join(FEATURES))

    mlflow.log_param(
        "train_period",
        f"{train_df['dteday'].min().date()} to "
        f"{train_df['dteday'].max().date()}"
    )

    mlflow.log_param(
        "test_period",
        f"{test_df['dteday'].min().date()} to "
        f"{test_df['dteday'].max().date()}"
    )

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)

    # Log trained model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=X_train.head(3)
    )

print(f"Linear Regression MAE: {mae:.2f}")
print(f"Linear Regression RMSE: {rmse:.2f}")
