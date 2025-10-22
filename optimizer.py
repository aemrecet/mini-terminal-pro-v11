import numpy as np, pandas as pd
def mean_variance_weights(returns: pd.DataFrame):
    mu = returns.mean()
    cov = returns.cov() + 1e-6*np.eye(len(mu))
    try:
        inv = np.linalg.pinv(cov.values)
        w = inv @ mu.values
        w = w / np.sum(np.abs(w))
    except Exception:
        w = np.ones(len(mu))/len(mu)
    return pd.Series(w, index=returns.columns)
def risk_parity_weights(returns: pd.DataFrame, iters=500, lr=0.01):
    cov = returns.cov() + 1e-8*np.eye(returns.shape[1])
    n = returns.shape[1]
    w = np.ones(n)/n
    for _ in range(iters):
        port_vol = np.sqrt(w @ cov.values @ w)
        if port_vol <= 0: break
        mrc = (cov.values @ w)/port_vol
        rc = w * mrc
        target = port_vol / n
        grad = rc - target
        w = w - lr*grad
        w = np.clip(w, 0, None)
        s = w.sum()
        if s > 0: w = w / s
    return pd.Series(w, index=returns.columns)
