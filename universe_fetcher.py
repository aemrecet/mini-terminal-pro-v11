import os, pandas as pd, requests
from dotenv import load_dotenv
load_dotenv()
BIST_CSV = "tickers/bist_full.csv"
SP500_CSV = "tickers/sp500.csv"
NASDAQ_CSV = "tickers/nasdaq_all.csv"
def ensure_dirs():
    os.makedirs("tickers", exist_ok=True)
def fetch_bist_all(write_csv=True):
    api = os.getenv("FINNHUB_API_KEY")
    if not api:
        raise RuntimeError("FINNHUB_API_KEY yok (.env veya Streamlit Secrets).")
    from finnhub import Client
    c = Client(api_key=api)
    data = c.stock_symbols('BIST')
    rows = []
    for d in data:
        sym = (d.get('symbol') or '').strip()
        name = (d.get('description') or '').strip()
        typ = (d.get('type') or '').lower()
        if typ in ("common stock","equity","") and sym:
            sym_yf = sym if sym.endswith(".IS") else (sym + ".IS")
            rows.append((sym_yf, name, 'Unknown'))
    df = pd.DataFrame(rows, columns=['symbol','name','sector']).drop_duplicates('symbol')
    if write_csv:
        ensure_dirs(); df.to_csv(BIST_CSV, index=False)
    return df
def fetch_sp500_wiki(write_csv=True):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0][['Symbol','Security','GICS Sector']]
    df.columns = ['symbol','name','sector']
    if write_csv:
        ensure_dirs(); df.to_csv(SP500_CSV, index=False)
    return df
def fetch_nasdaq_all(write_csv=True):
    base = "https://www.nasdaqtrader.com/dynamic/SymDir/"
    urls = ["nasdaqlisted.txt","otherlisted.txt"]
    frames = []
    for u in urls:
        r = requests.get(base+u, timeout=30)
        lines = [line for line in r.text.splitlines() if '|' in line]
        header = lines[0].split('|')
        data = [line.split('|') for line in lines[1:]]
        df = pd.DataFrame(data, columns=header)
        if 'Symbol' in df.columns and 'Security Name' in df.columns:
            frames.append(df[['Symbol','Security Name']].rename(columns={'Symbol':'symbol','Security Name':'name'}))
    all_df = pd.concat(frames).drop_duplicates('symbol')
    all_df['sector'] = "Unknown"
    if write_csv:
        ensure_dirs(); all_df.to_csv(NASDAQ_CSV, index=False)
    return all_df
def cached(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
def get_cached_universes():
    return {'bist': cached(BIST_CSV), 'sp500': cached(SP500_CSV), 'nasdaq': cached(NASDAQ_CSV)}
