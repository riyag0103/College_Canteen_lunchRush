FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/champion_mlflow_model/ ./models/champion_mlflow_model/

EXPOSE 8000

CMD ["uvicorn", "src.canteen_api.main:app", "--host", "0.0.0.0", "--port", "8000"]