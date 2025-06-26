# predict/predict_latest.py

import pandas as pd
import joblib
import mlflow
import lightgbm as lgb
from pathlib import Path

def predict_for_stock(ticker):
    model_uri = f"runs:/{get_latest_run_id(ticker)}/model"
    model = mlflow.sklearn.load_model(model_uri)

    df = pd.read_csv(f"data/processed/{ticker}.csv")
    df = df.dropna()

    latest = df.iloc[-1:]
    features = ['rsi', 'macd', 'bb_high', 'bb_low', 'bb_width']
    X_latest = latest[features]

    pred = model.predict(X_latest)[0]
    proba = model.predict_proba(X_latest)[0][1]  # Prob of going up

    return {
        "predicted_class": int(pred),
        "confidence": round(proba, 2),
        "latest_close": latest["close"].values[0],
        "date": latest["date"].values[0]
    }

def get_latest_run_id(ticker):
    # This will read from MLflow logs and get latest run for ticker
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string=f"params.ticker = '{ticker}'",
        order_by=["start_time DESC"]
    )
    return runs[0].info.run_id if runs else None
