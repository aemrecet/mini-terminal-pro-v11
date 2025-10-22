from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_fetcher import fetch_company_news
import numpy as np, pandas as pd
vader = SentimentIntensityAnalyzer()
def news_sentiment_score(symbol: str, hours: int = 6):
    df = fetch_company_news(symbol, days=2)
    if df is None or df.empty:
        return 0.0, pd.DataFrame()
    cutoff = pd.Timestamp.utcnow().tz_localize('UTC') - pd.Timedelta(hours=hours)
    df = df[df['datetime'] >= cutoff]
    scores, rows = [], []
    for _, r in df.iterrows():
        t = f"{r.get('headline','')} {r.get('summary','')}"
        if any(tok in t.lower() for tok in [' şirket','hisse','borsa','türkiye','tl']):
            try:
                t = GoogleTranslator(source='auto', target='en').translate(t)
            except Exception:
                pass
        s = vader.polarity_scores(t)['compound']
        scores.append(s)
        rows.append({'datetime': r['datetime'], 'headline': r.get('headline',''), 'score': s, 'url': r.get('url')})
    score = float(np.tanh(np.mean(scores))) if scores else 0.0
    return score, pd.DataFrame(rows).sort_values('datetime', ascending=False)
def combine_signal(prob_up: float, senti: float, cfg: dict) -> float:
    return float(cfg['signal']['model_weight']*prob_up + cfg['signal']['sentiment_weight']*(0.5*(senti+1.0)))
