import pandas as pd
from binance_fetcher import get_all_symbols, get_klines
from portfolio_backtest import backtest_multi
def build_returns_matrix(quote="USDT", interval="1d", limit=365, top_n=30):
    syms = get_all_symbols(quote=quote, spot_only=True)[:top_n]
    rets = []
    for s in syms:
        try:
            df = get_klines(s, interval=interval, limit=limit)
            r = df['Close'].pct_change().rename(s)
            rets.append(r)
        except Exception:
            continue
    if not rets: return pd.DataFrame(), []
    R = pd.concat(rets, axis=1).dropna()
    return R, R.columns.tolist()
def optimize_and_backtest(R, method='equal', rebalance='daily', fee_bp=10, slippage_bp=5):
    return backtest_multi(R, method=method, rebalance=rebalance, fee_bp=fee_bp, slippage_bp=slippage_bp)
