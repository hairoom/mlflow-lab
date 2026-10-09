
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# ==========================================
# 1. Configuration
# ==========================================

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Bike_Sharing_Forecasting"
REGISTERED_MODEL_NAME = "Capital_Bikeshare_Demand"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


# ==========================================
# 2. Retrieve Experiment Runs
# ==========================================

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError("Experiment not found.")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    max_results=50000
)

if runs.empty or "metrics.MAE" not in runs.columns:
    raise ValueError("No completed runs with MAE found.")

runs = runs.dropna(subset=["metrics.MAE"])

# ==========================================
# 3. Retrieve Baseline
# ==========================================

baseline_runs = runs[
    runs["tags.mlflow.runName"] == "Baseline_Yesterday"
]

if baseline_runs.empty:
    raise ValueError("Baseline run not found.")

# Use latest baseline
baseline = baseline_runs.sort_values(
    "start_time", ascending=False
).iloc[0]

baseline_mae = float(baseline["metrics.MAE"])

print(f"Baseline MAE: {baseline_mae:.2f}")


# ==========================================
# 4. Find Best Candidate
# ==========================================

candidates = runs[
    runs["tags.mlflow.runName"].isin([
        "Linear_Regression",
        "Random_Forest"
    ])
].copy()

if candidates.empty:
    raise ValueError("No candidate models found.")

# Only consider models better than baseline
candidates = candidates[
    candidates["metrics.MAE"] < baseline_mae
]

if candidates.empty:
    print("No candidate beats the baseline.")
    raise SystemExit(0)

best = candidates.sort_values(
    "metrics.MAE", ascending=True
).iloc[0]

best_run_id = best["run_id"]
best_mae = float(best["metrics.MAE"])
best_name = best["tags.mlflow.runName"]

print(f"Best candidate: {best_name}")
print(f"Candidate MAE: {best_mae:.2f}")


# ==========================================
# 5. Check Existing Champion
# ==========================================

try:
    champion = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion"
    )

except MlflowException as e:
    # Only treat a missing model or alias as
    # the first champion selection.
    if getattr(e, "error_code", None) not in (
        "RESOURCE_DOES_NOT_EXIST",
    ):
        raise

    champion = None

if champion is not None:

    champion_run_id = champion.run_id
    champion_run = client.get_run(champion_run_id)

    champion_mae = champion_run.data.metrics.get("MAE")

    if champion_mae is None:
        raise ValueError(
            "Current champion has no recorded MAE."
        )

    print(f"Current Champion Version: {champion.version}")
    print(f"Current Champion MAE: {champion_mae:.2f}")

    # Same run: no need to register again
    if best_run_id == champion_run_id:
        print("Best candidate is already Champion.")
        print("No update required.")
        raise SystemExit(0)

    # Worse or equal performance: keep champion
    if best_mae >= champion_mae:
        print("No improvement over current Champion.")
        print("Champion remains unchanged.")
        raise SystemExit(0)

else:
    print("No existing Champion found.")


# ==========================================
# 6. Retrieve Logged Model
# ==========================================

logged_models = client.search_logged_models(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"source_run_id = '{best_run_id}'"
)

if not logged_models:
    raise ValueError(
        "No logged model found for the selected run."
    )

model_uri = f"models:/{logged_models[0].model_id}"


# ==========================================
# 7. Register New Model Version
# ==========================================

registered = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME,
    await_registration_for=300
)

# Record evaluation metadata on model version
client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered.version,
    key="validation_mae",
    value=str(best_mae)
)

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered.version,
    key="source_run_id",
    value=best_run_id
)


# ==========================================
# 8. Update Champion Alias
# ==========================================

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="champion",
    version=registered.version
)

print("\nChampion updated successfully!")
print(f"Model: {REGISTERED_MODEL_NAME}")
print(f"Version: {registered.version}")
print(f"Source Run: {best_run_id}")
print(f"MAE: {best_mae:.2f}")
