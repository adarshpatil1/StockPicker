# trainer/train_classifier.py

import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

import os
from pathlib import Path

def train_on_ticker(ticker):
    print(f"📊 Training model for {ticker}")
    file_path = Path(f"data/processed/{ticker}")
    df = pd.read_csv(file_path)
    df = df.dropna()

    features = ['rsi', 'macd', 'bb_high', 'bb_low', 'bb_width']
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    with mlflow.start_run(run_name=ticker):
        mlflow.log_param("ticker", ticker)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        mlflow.sklearn.log_model(model, "model")

        # === Plots === #
        # Confusion matrix
        plt.figure(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {ticker}")
        plt.tight_layout()
        cm_path = f"conf_matrix_{ticker}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve: {ticker}")
        plt.legend()
        roc_path = f"roc_curve_{ticker}.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)

        # Feature importance
        plt.figure(figsize=(6, 4))
        lgb.plot_importance(model, max_num_features=10)
        plt.title(f"Feature Importance: {ticker}")
        plt.tight_layout()
        feat_path = f"feat_importance_{ticker}.png"
        plt.savefig(feat_path)
        mlflow.log_artifact(feat_path)

        print(f"✅ {ticker} logged to MLflow: ACC={acc:.3f} F1={f1:.3f} AUC={roc:.3f}")

def train_all():
    processed_dir = Path("data/processed")
    csv_files = list(processed_dir.glob("*.csv"))

    for file in csv_files:
        ticker = file.name
        try:
            train_on_ticker(ticker)
        except Exception as e:
            print(f"❌ Error training {ticker}: {e}")

if __name__ == "__main__":
    train_all()
