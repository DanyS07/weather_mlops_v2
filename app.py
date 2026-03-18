import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(layout="wide")

# -----------------------------------
# Ensure pipeline
# -----------------------------------
def ensure_pipeline():
    if not os.path.exists("models/technopark_model.pkl"):
        st.warning("⚙️ Running pipeline...")
        os.system("pip install -r requirements.txt")
        os.system("dvc repro")

ensure_pipeline()

# -----------------------------------
# Load version
# -----------------------------------
if os.path.exists("version.json"):
    version = json.load(open("version.json"))
else:
    version = {"version": "N/A", "trained_on": "N/A", "rmse_technopark": 0.0}

# -----------------------------------
# SIDEBAR (Quick stats placeholder)
# -----------------------------------
st.sidebar.title("📊 Quick Stats")

# -----------------------------------
# HEADER
# -----------------------------------
st.title("🌦 Weather Forecast Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Model Version", version.get("version", "N/A"))
col2.metric("Last Trained", str(version.get("trained_on", "N/A")).split(".")[0])
col3.metric("RMSE", f"{version.get('rmse_technopark', 0):.2f} °C")

st.markdown("---")

# -----------------------------------
# Prediction + actual data
# -----------------------------------
def get_data(region):
    try:
        model = joblib.load(f"models/{region}_model.pkl")
        scaler = joblib.load(f"models/scaler_{region}.pkl")

        X = np.load(f"data/processed/X_test_{region}.npy")

        latest = X[-1].reshape(1, -1)
        pred = model.predict(latest)[0]

        # inverse scaling
        dummy = np.zeros((len(pred), scaler.n_features_in_))
        dummy[:, 0] = pred
        forecast = scaler.inverse_transform(dummy)[:, 0]

        # recent actuals (last 48 hrs)
        actual = scaler.inverse_transform(X[-48:, :])[:, 0]

        return forecast, actual

    except Exception as e:
        st.error(f"Error: {e}")
        return np.zeros(24), np.zeros(48)

# -----------------------------------
# DISPLAY FUNCTION
# -----------------------------------
def display(region, title):

    st.subheader(title)

    forecast, actual = get_data(region)

    # Forecast DF
    forecast_df = pd.DataFrame({
        "Hour": list(range(1, 25)),
        "Forecast": forecast
    })

    # Actual DF
    actual_df = pd.DataFrame({
        "Hour": list(range(-47, 1)),
        "Actual": actual
    })

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Temp", f"{forecast.max():.1f} °C")
    c2.metric("Min Temp", f"{forecast.min():.1f} °C")
    c3.metric("Avg Temp", f"{forecast.mean():.1f} °C")

    # Sidebar update
    st.sidebar.markdown(f"### {region.title()}")
    st.sidebar.write(f"Max: {forecast.max():.1f} °C")
    st.sidebar.write(f"Min: {forecast.min():.1f} °C")
    st.sidebar.write(f"Avg: {forecast.mean():.1f} °C")

    # Combine for overlay chart
    combined = pd.concat([
        actual_df.set_index("Hour"),
        forecast_df.set_index("Hour")
    ], axis=1)

    st.markdown("### 📈 Forecast vs Recent Actuals")
    st.line_chart(combined)

    with st.expander("📊 Detailed Data"):
        st.dataframe(combined)

# -----------------------------------
# TABS
# -----------------------------------
tab1, tab2 = st.tabs(["Technopark", "Thampanoor"])

with tab1:
    display("technopark", "Technopark - Next 24 Hour Forecast")

with tab2:
    display("thampanoor", "Thampanoor - Next 24 Hour Forecast")

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.caption("MLOps Weather Forecast | Random Forest | Streamlit + DVC")