import logging
import json
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI

from .schemas import PredictionRequest, PredictionResponse
from .model_loader import model


app = FastAPI(
    title="College Canteen Demand Prediction API",
    version="1.0"
)

logger = logging.getLogger("prediction_api")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(handler)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    input_data = pd.DataFrame([request.model_dump()])

    prediction = float(model.predict(input_data)[0])

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "prediction",
        "prediction": prediction
    }

    logger.info(json.dumps(log_entry))

    return {"prediction": prediction}