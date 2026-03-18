import numpy as np, json, yaml, joblib, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

params = yaml.safe_load(open("params.yaml"))

def train(r):
    Xtr = np.load(f"data/processed/X_train_{r}.npy")
    ytr = np.load(f"data/processed/y_train_{r}.npy")
    Xte = np.load(f"data/processed/X_test_{r}.npy")
    yte = np.load(f"data/processed/y_test_{r}.npy")

    m = RandomForestRegressor(**params["model"], n_jobs=-1)
    m.fit(Xtr,ytr)

    p = m.predict(Xte)

    mae = mean_absolute_error(yte,p)
    rmse = mean_squared_error(yte,p)**0.5

    joblib.dump(m, f"models/{r}_model.pkl")
    return mae, rmse

os.makedirs("models", exist_ok=True)

mae1, rmse1 = train("technopark")
mae2, rmse2 = train("thampanoor")

json.dump({
    "rmse_technopark": rmse1,
    "rmse_thampanoor": rmse2
}, open("metrics.json","w"), indent=4)

json.dump({
    "version": datetime.now().strftime("%Y%m%d"),
    "trained_on": str(datetime.now()),
    "rmse_technopark": rmse1,
    "rmse_thampanoor": rmse2
}, open("version.json","w"), indent=4)