import requests, pandas as pd
BINANCE_BASE = "https://api.binance.com"
def get_all_symbols(quote="USDT", spot_only=True):
    j = requests.get(f"{BINANCE_BASE}/api/v3/exchangeInfo", timeout=20).json()
    syms = []
    for s in j['symbols']:
        if spot_only and s.get('isSpotTradingAllowed') != True: continue
        if s['quoteAsset'] == quote and s['status'] == 'TRADING': syms.append(s['symbol'])
    return sorted(list(set(syms)))
def get_klines(symbol, interval="1d", limit=365):
    j = requests.get(f"{BINANCE_BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}", timeout=30).json()
    cols = ['open_time','Open','High','Low','Close','Volume','close_time','qav','trades','taker_base','taker_quote','ignore']
    df = pd.DataFrame(j, columns=cols)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('open_time')
    for c in ['Open','High','Low','Close','Volume']: df[c] = df[c].astype(float)
    return df[['Open','High','Low','Close','Volume']]
