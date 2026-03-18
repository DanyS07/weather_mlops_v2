import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import os
os.system("pip install requests")

st.set_page_config(layout="wide")

# -----------------------------------
# Ensure pipeline runs (robust version)
# -----------------------------------
def ensure_pipeline():
    if not os.path.exists("models/technopark_model.pkl"):
        st.warning("⚙️ First-time setup: Running ML pipeline...")

        # Install dependencies (important for Streamlit Cloud)
        os.system("pip install -r requirements.txt")

        # Run pipeline
        exit_code = os.system("dvc repro")

        if exit_code != 0:
            st.error("❌ Pipeline failed. Check logs in Streamlit.")
            st.stop()


ensure_pipeline()

# -----------------------------------
# Load version info safely
# -----------------------------------
if os.path.exists("version.json"):
    version = json.load(open("version.json"))
else:
    version = {
        "version": "N/A",
        "trained_on": "N/A",
        "rmse_technopark": 0.0,
        "rmse_thampanoor": 0.0
    }

# -----------------------------------
# UI HEADER
# -----------------------------------
st.title("🌦 Weather Forecast Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Model Version", version.get("version", "N/A"))
col2.metric("Last Trained", str(version.get("trained_on", "N/A")).split(".")[0])
col3.metric("RMSE (Technopark)", f"{version.get('rmse_technopark', 0):.2f} °C")

st.markdown("---")

# -----------------------------------
# Prediction function
# -----------------------------------
def get_prediction(region):
    try:
        model = joblib.load(f"models/{region}_model.pkl")
        scaler = joblib.load(f"models/scaler_{region}.pkl")

        X = np.load(f"data/processed/X_test_{region}.npy")
        latest = X[-1].reshape(1, -1)

        pred = model.predict(latest)[0]

        # Inverse scaling
        dummy = np.zeros((len(pred), scaler.n_features_in_))
        dummy[:, 0] = pred
        real_values = scaler.inverse_transform(dummy)[:, 0]

        return real_values

    except Exception as e:
        st.error(f"❌ Error loading model for {region}: {e}")
        return np.zeros(24)


# -----------------------------------
# Display function
# -----------------------------------
def display(region, title):
    st.subheader(title)

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

    st.markdown("### 📈 Next 24 Hours Forecast")
    st.line_chart(df.set_index("Hour"))

    with st.expander("📊 View Data Table"):
        st.dataframe(df)


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