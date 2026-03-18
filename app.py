import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(layout="wide")

# -----------------------------
# Ensure model exists
# -----------------------------
if not os.path.exists("models"):
    os.system("dvc repro")

# -----------------------------
# Load metadata
# -----------------------------
version = json.load(open("version.json"))

st.title("🌦 Weather Forecast Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Model Version", version["version"])
col2.metric("Last Trained", version["trained_on"].split(".")[0])
col3.metric("RMSE", f"{version['rmse_technopark']:.2f} °C")

st.markdown("---")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Technopark", "Thampanoor"])


# -----------------------------
# Prediction function
# -----------------------------
def get_prediction(region):
    model = joblib.load(f"models/{region}_model.pkl")
    scaler = joblib.load(f"models/scaler_{region}.pkl")

    X = np.load(f"data/processed/X_test_{region}.npy")
    latest = X[-1].reshape(1, -1)

    pred = model.predict(latest)[0]

    # inverse scaling
    dummy = np.zeros((len(pred), scaler.n_features_in_))
    dummy[:, 0] = pred
    real_values = scaler.inverse_transform(dummy)[:, 0]

    return real_values


def display(region):
    pred = get_prediction(region)

    df = pd.DataFrame({
        "Hour": list(range(1, 25)),
        "Temperature (°C)": pred
    })

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Temp", f"{df['Temperature (°C)'].max():.1f} °C")
    c2.metric("Min Temp", f"{df['Temperature (°C)'].min():.1f} °C")
    c3.metric("Avg Temp", f"{df['Temperature (°C)'].mean():.1f} °C")

    st.markdown("### Next 24 Hours Forecast")
    st.line_chart(df.set_index("Hour"))

    with st.expander("View Data"):
        st.dataframe(df)


# -----------------------------
# Tabs content
# -----------------------------
with tab1:
    st.subheader("Technopark Forecast")
    display("technopark")

with tab2:
    st.subheader("Thampanoor Forecast")
    display("thampanoor")