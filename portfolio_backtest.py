import pandas as pd, numpy as np
from optimizer import mean_variance_weights, risk_parity_weights
def rebal_dates(index, freq):
    if freq == 'daily': return index
    if freq == 'weekly': return index[index.weekday == 0]
    if freq == 'monthly': return index[index.is_month_start]
    return index
def backtest_multi(R: pd.DataFrame, method='equal', rebalance='daily', fee_bp=10, slippage_bp=5):
    R = R.dropna()
    dates = R.index
    rdates = rebal_dates(dates, rebalance)
    w = None; curve = []; last_w = None
    fee = fee_bp/10000.0; slp = slippage_bp/10000.0
    for t, d in enumerate(dates):
        if (d in rdates) or (w is None):
            if method == 'equal':
                w_new = pd.Series(1/len(R.columns), index=R.columns)
            elif method == 'meanvar':
                w_new = mean_variance_weights(R.loc[:d].tail(252))
            else:
                w_new = risk_parity_weights(R.loc[:d].tail(252))
            cost = ((w_new - last_w).abs().sum() if last_w is not None else 0.0) * (fee+slp)
            w = w_new; last_w = w_new
        day_ret = float((R.iloc[t] @ w))
        curve.append(day_ret - (cost if d in rdates else 0.0))
    equity = (1 + pd.Series(curve, index=dates)).cumprod()
    return equity, w
