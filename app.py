import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# -----------------------------------
# Ensure pipeline
# -----------------------------------
def ensure_pipeline():
    if not os.path.exists("models/technopark_model.pkl"):
        st.warning("⚙️ Running ML pipeline...")
        os.system("pip install -r requirements.txt")
        exit_code = os.system("dvc repro")

        if exit_code != 0:
            st.error("Pipeline failed. Check logs.")
            st.stop()

ensure_pipeline()

# -----------------------------------
# Load version
# -----------------------------------
if os.path.exists("version.json"):
    version = json.load(open("version.json"))
else:
    version = {"version": "N/A", "trained_on": "N/A", "rmse_technopark": 0.0}

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.title("📊 Quick Stats")

# -----------------------------------
# Header
# -----------------------------------
st.title("🌦 Weather Forecast Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Model Version", version.get("version", "N/A"))
col2.metric("Last Trained", str(version.get("trained_on", "N/A")).split(".")[0])
col3.metric("RMSE", f"{version.get('rmse_technopark', 0):.2f} °C")

st.markdown("---")

# -----------------------------------
# Data function (FIXED SCALING)
# -----------------------------------
def get_data(region):
    try:
        model = joblib.load(f"models/{region}_model.pkl")
        scaler = joblib.load(f"models/scaler_{region}.pkl")

        X = np.load(f"data/processed/X_test_{region}.npy")

        latest = X[-1].reshape(1, -1)
        pred = model.predict(latest)[0]

        # ---- Forecast inverse scaling ----
        dummy_pred = np.zeros((len(pred), scaler.n_features_in_))
        dummy_pred[:, 0] = pred
        forecast = scaler.inverse_transform(dummy_pred)[:, 0]

        # ---- Actual inverse scaling ----
        temp_scaled = X[-48:, 0].reshape(-1, 1)

        dummy_actual = np.zeros((len(temp_scaled), scaler.n_features_in_))
        dummy_actual[:, 0] = temp_scaled.flatten()
        actual = scaler.inverse_transform(dummy_actual)[:, 0]

        return forecast, actual

    except Exception as e:
        st.error(f"Error: {e}")
        return np.zeros(24), np.zeros(48)

# -----------------------------------
# Display function
# -----------------------------------
def display(region, title):

    st.subheader(title)

    forecast, actual = get_data(region)

    # Time axis (NO NEGATIVE VALUES)
    actual_hours = list(range(1, 49))
    forecast_hours = list(range(49, 73))


    # Sidebar update
    st.sidebar.markdown(f"### {region.title()}")
    st.sidebar.write(f"Max: {forecast.max():.1f} °C")
    st.sidebar.write(f"Min: {forecast.min():.1f} °C")
    st.sidebar.write(f"Avg: {forecast.mean():.1f} °C")

    # -----------------------------------
    # Plotly Chart (PROFESSIONAL)
    # -----------------------------------
    fig = go.Figure()

    # Actual (dark blue)
    fig.add_trace(go.Scatter(
        x=actual_hours,
        y=actual,
        mode='lines',
        name='Actual (Last 48h)',
        line=dict(color='royalblue', width=3)
    ))

    # Forecast (orange)
    fig.add_trace(go.Scatter(
        x=forecast_hours,
        y=forecast,
        mode='lines',
        name='Forecast (Next 24h)',
        line=dict(color='orange', width=3, dash='dash')
    ))

    fig.update_layout(
        title="Forecast vs Recent Actuals",
        xaxis_title="Time (Hours)",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        legend=dict(
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.6)"
            )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    with st.expander("📊 Detailed Data"):
        df_actual = pd.DataFrame({"Hour": actual_hours, "Actual": actual})
        df_forecast = pd.DataFrame({"Hour": forecast_hours, "Forecast": forecast})
        st.dataframe(pd.concat([df_actual, df_forecast]))

# -----------------------------------
# Tabs
# -----------------------------------
tab1, tab2 = st.tabs(["Technopark", "Thampanoor"])

with tab1:
    display("technopark", "Technopark - Next 24 Hour Forecast")

with tab2:
    display("thampanoor", "Thampanoor - Next 24 Hour Forecast")

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.caption("MLOps Weather Forecast | Random Forest | Streamlit + DVC")