# Capital Bikeshare Demand Forecasting with MLflow

## Project Overview

This project predicts **daily bike rental demand** for Capital Bikeshare in Washington, D.C. The goal is to estimate tomorrow's total rentals using information available the previous evening, helping operations teams plan staffing and fleet capacity.

The project compares two regression models against a simple benchmark and uses **MLflow** for experiment tracking, model comparison, and model versioning.

## Dataset

The [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) contains Capital Bikeshare rental records from 2011–2012:

- `day.csv`: 731 daily observations (**used for modeling**).
- `hour.csv`: 17,379 hourly observations (included for reference, not used).
- `Readme.txt`: original dataset description and variable definitions.

The prediction target is `cnt`, the total number of daily rentals.

### Features

| Feature | Description |
| --- | --- |
| `season` | Season of the forecast day |
| `mnth` | Month of the forecast day |
| `weekday` | Day of the week |
| `holiday` | Holiday indicator |
| `workingday` | Working-day indicator |
| `yesterday_cnt` | Rental count from the previous day (lag 1) |
| `last_week_cnt` | Rental count from seven days earlier (lag 7) |

Observed weather fields are excluded because tomorrow's actual weather is unavailable the previous evening. The target's same-day components (`casual` and `registered`) are also excluded to prevent data leakage.

## Models

| Script | Approach | Training required? |
| --- | --- | --- |
| `baseline.py` | Predict tomorrow's rentals using yesterday's count | No |
| `linear_regression.py` | Linear Regression | Yes |
| `random_forest.py` | Random Forest Regressor | Yes |

`prepare_data.py` handles shared preprocessing, lag-feature creation, and a chronological 80/20 train/test split. All three approaches are evaluated on the same test period using **MAE** and **RMSE**.

## Project Structure

```text
Bike-Sharing-MLflow/
├── data/
│   ├── day.csv
│   ├── hour.csv
│   └── Readme.txt
├── prepare_data.py
├── baseline.py
├── linear_regression.py
├── random_forest.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone and install dependencies

```bash
git clone <YOUR_REPOSITORY_URL>
cd Bike-Sharing-MLflow
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

On Windows, activate the virtual environment with `.venv\Scripts\activate`.

### 2. Start the MLflow tracking server

From the project root, run:

```bash
mkdir -p mlflow_storage
mlflow server \
  --backend-store-uri "sqlite:///$(pwd)/mlflow_storage/mlflow.db" \
  --artifacts-destination "$(pwd)/mlflow_storage/artifacts" \
  --port 5000
```

Open **http://127.0.0.1:5000** to access the MLflow UI. Keep this terminal running. The SQLite database and model artifacts are stored locally in `mlflow_storage/`.

### 3. Run the experiments

Open a second terminal in the project root, activate the same environment, and execute:

```bash
python prepare_data.py
python baseline.py
python linear_regression.py
python random_forest.py
```

The scripts log runs to the `Bike_Sharing_Forecasting` experiment at `http://127.0.0.1:5000`.

### 4. Compare results in MLflow

1. Open the **Bike_Sharing_Forecasting** experiment.
2. Select the Baseline, Linear Regression, and Random Forest runs.
3. Compare MAE and RMSE (lower is better).
4. Inspect parameters, feature lists, training/test periods, artifacts, and Run IDs.
5. If a trained model improves on the baseline, register it in the **Model Registry** under a name such as `Capital_Bikeshare_Demand` to create a retrievable version linked to its source run.

> The baseline is a prediction rule rather than a fitted estimator, so it is logged as an MLflow run without a trained model artifact.

## MLflow Tracking

Each model run records its settings, features, data periods, and evaluation metrics. Trained models are saved as MLflow model artifacts. The Model Registry can then be used to manage named model versions and trace them back to the training runs that produced them.

## Limitations and Future Improvements

- The current chronological 80/20 split is suitable for an initial demonstration. For model selection and tuning, use separate training, validation, and final test periods or rolling-origin validation.
- Calendar categories in Linear Regression can be encoded with one-hot encoding instead of treating their numeric codes as continuous quantities.
- Weather forecasts available the previous evening could be added in future work, but observed next-day weather must not be used as a predictor.
- The dataset covers 2011–2012 and does not represent current operating conditions.

## Dataset Attribution

Fanaee-T, H., & Gama, J. (2013). *Event labeling combining ensemble detectors for anomaly detection*. Progress in Artificial Intelligence. https://doi.org/10.1007/s13748-013-0040-3

Dataset source: [UCI Machine Learning Repository — Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
