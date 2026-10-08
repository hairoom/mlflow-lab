
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prepare_data import load_data, FEATURES

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Bike_Sharing_Forecasting")

# Load data
X_train, X_test, y_train, y_test, train_df, test_df = load_data()

# Model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

with mlflow.start_run(run_name="Random_Forest"):

    mlflow.set_tag("model_type", "Random Forest")
    mlflow.set_tag("analyst", "Analyst B")

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
    name="random_forest",
    input_example=X_train.head(3),
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
    

print(f"Random Forest MAE: {mae:.2f}")
print(f"Random Forest RMSE: {rmse:.2f}")
