import mlflow.pyfunc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "champion_mlflow_model"

model = mlflow.pyfunc.load_model(str(MODEL_PATH))

print("Champion model loaded successfully!")