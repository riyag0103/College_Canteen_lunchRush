import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
os_path = ROOT

TRAIN = ROOT / "data/processed/train_clean.csv"
RAW_TRAIN = ROOT / "data/raw/train.csv"
LATER = ROOT / "data/raw/later.csv"
HOLDOUT = ROOT / "data/raw/holdout_features.csv"
V1_PATH = ROOT / "models/v1_model.pkl"
FINDINGS = ROOT / "reports/findings.yaml"

NUMERIC = [
    "students_on_campus", "menu_price_rs", "temp_c",
    "exam_days_left", "special_dishes", "queue_minutes_yesterday"
]
FEATURES = NUMERIC + ["day_type", "block"]
TARGET = "plates_sold"

with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

train = pd.read_csv(TRAIN)
later = pd.read_csv(LATER)
holdout = pd.read_csv(HOLDOUT)

later["date"] = pd.to_datetime(later["date"])
later = later.sort_values("date").reset_index(drop=True)

split = int(len(later) * 0.70)
later_train = later.iloc[:split]
check = later.iloc[split:]

v1 = joblib.load(V1_PATH)

# PSI: cleaned train vs later, using 10 reference quantile bins
psi_values = {}
for col in NUMERIC:
    ref = train[col].dropna()
    new = later[col].dropna()

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        psi_values[col] = 0.0
        continue

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(ref, bins=edges)[0]
    new_counts = np.histogram(new, bins=edges)[0]

    ref_pct = np.maximum(ref_counts / ref_counts.sum(), 0.0001)
    new_pct = np.maximum(new_counts / new_counts.sum(), 0.0001)

    psi_values[col] = float(
        np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
    )

# V1 block-wise MAE on all later data
v1_later_pred = v1.predict(later[FEATURES])
later["v1_prediction"] = v1_later_pred
mae_by_group = {}
for group in ["A-Block", "B-Block", "Hostel"]:
    rows = later[later["block"] == group]
    mae_by_group[group] = float(
        mean_absolute_error(rows[TARGET], rows["v1_prediction"])
    )

# Train V2 on train + first 70% of later data
combined = pd.concat([train, later_train], ignore_index=True)
v2 = clone(v1)
v2.fit(combined[FEATURES], combined[TARGET])

v1_check = v1.predict(check[FEATURES])
v2_check = v2.predict(check[FEATURES])

mae_v1 = float(mean_absolute_error(check[TARGET], v1_check))
mae_v2 = float(mean_absolute_error(check[TARGET], v2_check))

promote = mae_v2 <= mae_v1 * (1 - 0.12)
decision = "promote" if promote else "keep"

# MLflow logging and registration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(params["mlflow"]["experiment_name"])
client = MlflowClient()

sha = hashlib.sha256(RAW_TRAIN.read_bytes()).hexdigest()

with mlflow.start_run(run_name="stage4_v2_retraining") as run:
    mlflow.log_param("data_sha256", sha)
    mlflow.log_param("training_rows", len(combined))
    mlflow.log_param("later_train_fraction", 0.70)
    mlflow.log_metric("mae", mae_v2)

    mlflow.sklearn.log_model(v2, artifact_path="model")
    run_id = run.info.run_id

registered = mlflow.register_model(
    f"runs:/{run_id}/model",
    params["mlflow"]["registered_model_name"]
)
v2_version = registered.version

# Set champion alias according to the promotion decision
champion_version = v2_version if promote else 1
client.set_registered_model_alias(
    params["mlflow"]["registered_model_name"],
    params["mlflow"]["champion_alias"],
    str(champion_version)
)

champion = v2 if promote else v1

# Holdout predictions
predictions = champion.predict(holdout[FEATURES])
out = pd.DataFrame({
    "record_id": holdout["record_id"],
    "prediction": predictions
})
out_dir = ROOT / "predictions"
out_dir.mkdir(exist_ok=True)
out.to_csv(out_dir / "holdout_predictions.csv", index=False)

# Update Stage 4 fields in findings.yaml
with open(FINDINGS, "r", encoding="utf-8") as f:
    findings = yaml.safe_load(f)

tracking = findings["tracking"]
tracking["experiment_name"] = params["mlflow"]["experiment_name"]
tracking["best_run_id"] = "741a40ff7b8c42d6a0467931bea30515"
tracking["registered_model_name"] = params["mlflow"]["registered_model_name"]
tracking["champion_version"] = int(champion_version)

monitoring = findings["monitoring"]
monitoring["psi"] = psi_values
monitoring["most_drifted_feature"] = max(psi_values, key=psi_values.get)
monitoring["mae_by_group"] = mae_by_group
monitoring["worst_served_group"] = max(mae_by_group, key=mae_by_group.get)
monitoring["mae_v1_on_check"] = mae_v1
monitoring["mae_v2_on_check"] = mae_v2
monitoring["decision"] = decision

with open(FINDINGS, "w", encoding="utf-8") as f:
    yaml.safe_dump(findings, f, sort_keys=False)

print("\nStage 4 complete!")
print("PSI:", psi_values)
print("MAE by block:", mae_by_group)
print("V1 check MAE:", mae_v1)
print("V2 check MAE:", mae_v2)
print("Decision:", decision)
print("Champion version:", champion_version)
print("Predictions saved:", out_dir / "holdout_predictions.csv")
print("Findings updated:", FINDINGS)
