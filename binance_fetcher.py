import time
import requests
import pandas as pd

# Primary and fallbacks (some regions/CDNs occasionally fail)
_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

def _safe_get_json(url, timeout=20):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "mini-terminal/1.0"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def get_all_symbols(quote="USDT", spot_only=True, max_retries=3, sleep_sec=0.8):
    """Return a list of tradable spot symbols for a given quote asset.
    Avoids KeyError by validating response shape and retrying across mirrors.
    """
    last_err = None
    for _ in range(max_retries):
        for base in _BASES:
            j = _safe_get_json(f"{base}/api/v3/exchangeInfo")
            syms = []
            if isinstance(j, dict) and isinstance(j.get("symbols"), list):
                for s in j["symbols"]:
                    try:
                        if quote and s.get("quoteAsset") != quote:
                            continue
                        if spot_only and not s.get("isSpotTradingAllowed", False):
                            continue
                        if s.get("status") != "TRADING":
                            continue
                        name = s.get("symbol")
                        if name:
                            syms.append(name)
                    except Exception:
                        continue
                if syms:
                    return sorted(list(set(syms)))
            # record last error-like payload for debug
            last_err = j
        time.sleep(sleep_sec)
    # If all attempts failed, return empty list (caller should handle gracefully)
    return []

def get_klines(symbol, interval="1d", limit=365):
    for base in _BASES:
        url = f"{base}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        j = _safe_get_json(url, timeout=30)
        if isinstance(j, list) and j:
            cols = ['open_time','Open','High','Low','Close','Volume','close_time','qav','trades','taker_base','taker_quote','ignore']
            try:
                df = pd.DataFrame(j, columns=cols)
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df = df.set_index('open_time')
                for c in ['Open','High','Low','Close','Volume']:
                    df[c] = df[c].astype(float)
                return df[['Open','High','Low','Close','Volume']]
            except Exception:
                continue
    return pd.DataFrame()
