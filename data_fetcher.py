import os, requests
import yfinance as yf
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

FH_KEY = os.getenv('FINNHUB_API_KEY')

def _yf_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    if s.endswith(".IS"):
        return s
    if s.startswith("BIST:"):
        return s.split("BIST:")[1] + ".IS"
    return s

def _fh_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    if s.endswith(".IS"):
        core = s[:-3]
        return f"BIST:{core}"
    return s

def fetch_history(symbol, start='2018-01-01', end=None, interval='1d'):
    try:
        ys = _yf_symbol(symbol)
        df = yf.download(ys, start=start, end=end, interval=interval, progress=False, threads=False, auto_adjust=False)
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={'Adj Close':'AdjClose'})
            return df
    except Exception:
        pass
    return pd.DataFrame()

def fetch_company_news(symbol, days=2):
    if not FH_KEY:
        return pd.DataFrame()
    from finnhub import Client
    client = Client(api_key=FH_KEY)
    to_date = datetime.utcnow().date()
    from_date = to_date - pd.Timedelta(days=days)
    try:
        fs = _fh_symbol(symbol)
        news = client.company_news(fs, _from=from_date.strftime('%Y-%m-%d'), to=to_date.strftime('%Y-%m-%d'))
        df = pd.DataFrame(news)
        if not df.empty and 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True)
        return df
    except Exception:
        return pd.DataFrame()
