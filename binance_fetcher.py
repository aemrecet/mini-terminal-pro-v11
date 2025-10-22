import time, requests, pandas as pd

# Binance aynaları (varsa)
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
    last = None
    for _ in range(max_retries):
        for base in _BASES:
            j = _safe_get_json(f"{base}/api/v3/exchangeInfo")
            if isinstance(j, dict) and isinstance(j.get("symbols"), list):
                syms = []
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
            last = j
        time.sleep(sleep_sec)
    # yoksa boş dön
    return []

def get_klines(symbol, interval="1d", limit=365):
    # Binance Kline
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

# ---------- CoinGecko Fallback ----------
# Binance boşsa buraya düşeriz: sembolü "BTCUSDT" -> "btc" (base) çevirip CoinGecko OHLC çekeriz.
_CG_COIN_LIST_CACHE = None

def _cg_coin_list():
    global _CG_COIN_LIST_CACHE
    if _CG_COIN_LIST_CACHE is not None:
        return _CG_COIN_LIST_CACHE
    try:
        j = _safe_get_json("https://api.coingecko.com/api/v3/coins/list?include_platform=false", timeout=30)
        if isinstance(j, list) and j:
            _CG_COIN_LIST_CACHE = j
            return j
    except Exception:
        pass
    _CG_COIN_LIST_CACHE = []
    return _CG_COIN_LIST_CACHE

def _base_from_symbol(binance_symbol, quote="USDT"):
    # BTCUSDT -> BTC
    s = (binance_symbol or "").upper()
    if s.endswith(quote):
        return s[:-len(quote)]
    return s

def _cg_find_id_by_symbol(sym_lower):
    # CoinGecko listesinde 'symbol' eşleşmesi (örn: btc, eth)
    for item in _cg_coin_list():
        try:
            if item.get("symbol","").lower() == sym_lower:
                return item.get("id")
        except Exception:
            continue
    return None

def cg_get_ohlc(binance_symbol, days=365):
    base = _base_from_symbol(binance_symbol, quote="USDT").lower()
    coin_id = _cg_find_id_by_symbol(base)
    if not coin_id:
        return pd.DataFrame()
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={min(days, 365)}"
    j = _safe_get_json(url, timeout=30)
    # OHLC format: [timestamp, open, high, low, close]
    if isinstance(j, list) and j:
        df = pd.DataFrame(j, columns=["ts","Open","High","Low","Close"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("ts")
        df["Volume"] = 0.0  # CG ohlc endpoint volumeyi vermiyor
        return df[["Open","High","Low","Close","Volume"]]
    return pd.DataFrame()
