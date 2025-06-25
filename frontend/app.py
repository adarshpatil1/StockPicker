# app/dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go
from predict.predict_latest import predict_for_stock

st.set_page_config(layout="wide")

st.title("📈 QuantML Dashboard - Stock Prediction for Tomorrow")

# Load stock list
csv_dir = "data/processed"
available_stocks = sorted([f for f in os.listdir(csv_dir) if f.endswith(".csv")])

# Select a stock
stock_file = st.selectbox("Select a stock", available_stocks)
ticker = stock_file.replace(".csv", "")

# Load data
df = pd.read_csv(f"{csv_dir}/{stock_file}")
df = df.dropna()
df["date"] = pd.to_datetime(df["date"])
latest = df.iloc[-1:]

# Prediction
with st.spinner("Predicting..."):
    result = predict_for_stock(stock_file)

# === Layout ===
col1, col2 = st.columns(2)

# Prediction Panel
with col1:
    st.subheader("🧠 Model Prediction")
    pred_label = "📈 Price will go UP" if result["predicted_class"] == 1 else "📉 Price will go DOWN"
    st.metric(label="Prediction for Tomorrow", value=pred_label, delta=f"{int(result['confidence']*100)}% confidence")
    st.metric(label="Latest Close Price", value=f"₹ {result['latest_close']:.2f}")
    st.caption(f"Prediction based on data up to: {result['date']}")

# Price Chart
with col2:
    st.subheader("📊 Historical Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode='lines', name='Close Price'))
    fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

# Technical Indicators
with st.expander("📉 Technical Indicators"):
    st.line_chart(df.set_index("date")[["rsi", "macd"]])
    st.area_chart(df.set_index("date")[["bb_low", "bb_high"]])

# Metrics Plots from MLflow artifacts
st.divider()
st.subheader("📂 Model Visualizations")
artifact_dir = "."

col3, col4, col5 = st.columns(3)
for name, col in zip(["conf_matrix", "roc_curve", "feat_importance"], [col3, col4, col5]):
    path = f"{name}_{stock_file}"
    if os.path.exists(path):
        col.image(path, caption=name.replace("_", " ").title(), use_column_width=True)
    else:
        col.warning(f"{name} plot not found for this stock.")

