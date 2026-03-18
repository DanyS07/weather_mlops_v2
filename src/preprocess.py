import pandas as pd, numpy as np, yaml, os, joblib
from sklearn.preprocessing import MinMaxScaler

params = yaml.safe_load(open("params.yaml"))
L, H = params["features"]["lookback"], params["features"]["horizon"]

def process(region):
    df = pd.read_csv(f"data/raw/{region}.csv").dropna()

    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    df["day"] = pd.to_datetime(df["time"]).dt.dayofweek

    features = ["temp","humidity","precip","wind","hour","day"]

    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[features])

    X, y = [], []

    for i in range(len(data)-L-H):
        X.append(data[i:i+L].flatten())
        y.append(data[i+L:i+L+H,0])

    return np.array(X), np.array(y), scaler

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

for r in ["technopark","thampanoor"]:
    X,y,s = process(r)
    split = int(0.8*len(X))

    np.save(f"data/processed/X_train_{r}.npy", X[:split])
    np.save(f"data/processed/y_train_{r}.npy", y[:split])
    np.save(f"data/processed/X_test_{r}.npy", X[split:])
    np.save(f"data/processed/y_test_{r}.npy", y[split:])

    joblib.dump(s, f"models/scaler_{r}.pkl")