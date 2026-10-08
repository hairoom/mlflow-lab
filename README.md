# Capital Bikeshare Demand Forecasting with MLflow

This project predicts **daily bike rental demand for the next day** using historical Capital Bikeshare data. It compares a simple baseline with Linear Regression and Random Forest, while using **MLflow** to track experiments, compare metrics, and register a selected model.

The forecast is intended to be made the previous evening, so model inputs are limited to calendar information and rental counts already known at that time.

## Project Structure

```text
mlflow-lab/
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

Only `data/day.csv` is used for modelling. `hour.csv` is included as part of the original dataset but is not needed for this daily forecasting task.

## Setup

From the project root, install dependencies:

```bash
pip install -r requirements.txt
```

Create a local folder for MLflow storage and start the server:

```bash
mkdir -p mlflow_storage
mlflow server \
  --backend-store-uri "sqlite:///$(pwd)/mlflow_storage/mlflow.db" \
  --artifacts-destination "$(pwd)/mlflow_storage/artifacts" \
  --port 5000
```

Keep this terminal running. Open **http://127.0.0.1:5000** in your browser. Use a **second terminal**, from the project root, for the Python commands below.

> The scripts connect to `http://127.0.0.1:5000` and use the MLflow experiment `Bike_Sharing_Forecasting`.

## Step 1 — Prepare the Data

**Run:** `prepare_data.py`

```bash
python prepare_data.py
```

This script:

- Loads `data/day.csv` and sorts observations by date.
- Creates `yesterday_cnt` (previous day's rentals) and `last_week_cnt` (rentals seven days earlier).
- Uses calendar features (`season`, `mnth`, `weekday`, `holiday`, `workingday`) together with these two lag features.
- Drops rows without the required historical counts.
- Splits the data chronologically into **80% training** and **20% testing**.
- Prints the dataset shapes and the training/testing date ranges.

**Output:** The processed data is returned by `load_data()` for use in the other scripts; this step does not create a separate processed CSV or MLflow run.

## Step 2 — Evaluate the Baseline

**Run:** `baseline.py`

```bash
python baseline.py
```

The baseline predicts each day's rental count using **the previous day's actual rentals**:

```text
predicted_count_today = actual_count_yesterday
```

No model fitting is required. The script evaluates the baseline on the test period using **MAE** and **RMSE**, prints both scores, and logs a run named `Baseline_Yesterday` in MLflow.

**Check in MLflow UI:** Open the `Bike_Sharing_Forecasting` experiment and find the `Baseline_Yesterday` run. Its metrics provide the benchmark that trained models should beat.

## Step 3 — Train Linear Regression

**Run:** `linear_regression.py`

```bash
python linear_regression.py
```

This script loads the same training and testing data, fits a Linear Regression model, makes predictions on the test period, and calculates **MAE** and **RMSE**.

**Check in MLflow UI:** Find the `Linear_Regression` run. It records the model parameters, input features, training/testing periods, evaluation metrics, and the saved model artifact.

## Step 4 — Train Random Forest

**Run:** `random_forest.py`

```bash
python random_forest.py
```

This script trains a Random Forest regressor on the same split, initially using `n_estimators=100`, `max_depth=10`, and `random_state=42`. It evaluates predictions with **MAE** and **RMSE**.

**Check in MLflow UI:** Find the `Random_Forest` run. As with Linear Regression, its parameters, features, data periods, metrics, and model artifact are recorded. You can change model settings and rerun the script to create another experiment run.

## Step 5 — Compare Experiments in MLflow UI

1. Open **http://127.0.0.1:5000**.
2. Select the **`Bike_Sharing_Forecasting`** experiment.
3. Find the three runs: `Baseline_Yesterday`, `Linear_Regression`, and `Random_Forest`.
4. Select the runs and use the comparison view to inspect **MAE** and **RMSE** (lower is better).
5. Open individual runs to review their **Parameters**, **Metrics**, **Artifacts**, and **Run ID**.

All three runs use the same test period, making the error scores directly comparable. The baseline has no trained model artifact because it uses a fixed forecasting rule.

## Step 6 — Register a Selected Model

After comparing the results, register a trained model **only if it improves on the baseline** on the chosen evaluation metric.

1. Open the winning **Linear Regression** or **Random Forest** run in MLflow UI.
2. Locate the logged model and choose **Register Model** (the exact UI wording may vary by MLflow version).
3. Create a registered model named `Capital_Bikeshare_Demand`, or select that name if it already exists.
4. Confirm that a model **version** was created and that it links back to the source **Run ID**.

A registered model version can later be retrieved using a URI such as:

```python
import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/Capital_Bikeshare_Demand/1")
```

Replace `1` with the actual registered version. Registration is **not** the same as deploying the model.

## Quick Run Order

With the MLflow server already running in another terminal:

```bash
python prepare_data.py
python baseline.py
python linear_regression.py
python random_forest.py
```

Then open the MLflow UI to compare runs and register a selected model.

## Dataset and Forecasting Limitations

The dataset contains Capital Bikeshare rentals from **2011–2012**, aggregated by day (`day.csv`) and hour (`hour.csv`). The target is `cnt`, the total daily rental count. The original dataset documentation is included in `data/Readme.txt`.

Actual weather observed on the target day (`temp`, `atemp`, `hum`, `windspeed`, `weathersit`) is **not used**, because it would not be available the previous evening. Forecast weather could be added in a future version if historical forecasts were available.

This project is an initial experiment-tracking workflow. For more rigorous model selection, use chronological **train/validation/test** splits or rolling-origin validation, and reserve an untouched test period for the final baseline comparison. Categorical calendar features can also be encoded more appropriately for Linear Regression in a future iteration.

## Reference

Fanaee-T, H., & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence. https://doi.org/10.1007/s13748-013-0040-3
