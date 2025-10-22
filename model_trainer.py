import os
import joblib
import numpy as np
import pandas as pd
USE_LGB = True
try:
    import lightgbm as lgb
except Exception:
    USE_LGB = False
    from sklearn.ensemble import GradientBoostingClassifier
from data_fetcher import fetch_history
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df is None or df.empty:
        return pd.DataFrame()
    df["return"] = df["Close"].pct_change()
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["vol21"] = df["return"].rolling(21).std()
    df["mom5"] = df["Close"].pct_change(5)
    df = df.dropna()
    return df
def _features_and_target(df: pd.DataFrame, cfg: dict):
    feat = make_features(df)
    if feat.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    tgt = (feat["Close"].pct_change().shift(-1) > cfg["data"]["target_threshold"]).astype(int)
    feat = feat.dropna()
    y = tgt.loc[feat.index]
    X = feat[cfg["model"]["features"]]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    return X, y
def train_symbol(symbol: str, cfg: dict):
    start = cfg["data"]["start_date"]
    df = fetch_history(symbol, start=start)
    if df is None or df.empty:
        raise RuntimeError(f"Veri yok: {symbol}")
    X, y = _features_and_target(df, cfg)
    if X.empty or y.empty or len(np.unique(y)) < 2:
        raise RuntimeError(f"Eğitim için yeterli hedef yok: {symbol}")
    model_path = os.path.join(MODEL_DIR, f"model_{symbol.replace('/','_')}.pkl")
    if USE_LGB:
        params = cfg["model"]["lgb_params"]
        dtrain = lgb.Dataset(X, label=y)
        model = lgb.train(params, dtrain, num_boost_round=cfg["model"]["num_boost_round"])
        joblib.dump(model, model_path)
    else:
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, model_path)
    return model_path
def load_ensemble():
    if not os.path.isdir(MODEL_DIR):
        return []
    models = []
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".pkl"):
            fpath = os.path.join(MODEL_DIR, fname)
            try:
                models.append(joblib.load(fpath))
            except Exception:
                pass
    return models
def predict_proba_ensemble(models, X_row: pd.DataFrame):
    if not models or X_row is None or X_row.empty:
        return None
    preds = []
    for m in models:
        try:
            p = m.predict(X_row); p = float(p[0])
        except Exception:
            if hasattr(m, "predict_proba"):
                p = float(m.predict_proba(X_row)[0, 1])
            else:
                import numpy as np
                s = float(m.decision_function(X_row)[0])
                p = 1 / (1 + np.exp(-s))
        preds.append(p)
    return float(np.mean(preds)) if preds else None
