# analysis/sentiment.py
import requests
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from dotenv import load_dotenv

try:
    import streamlit as st
    NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")
except:
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

analyzer = SentimentIntensityAnalyzer()

# Stock name mappings for better news search
STOCK_NAMES = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'TCS Tata Consultancy',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'WIPRO.NS': 'Wipro',
    'ICICIBANK.NS': 'ICICI Bank',
    'SBIN.NS': 'State Bank India SBI',
    'ADANIENT.NS': 'Adani Enterprises',
    'TATAMOTORS.NS': 'Tata Motors',
    'BAJFINANCE.NS': 'Bajaj Finance',
}


def fetch_news(symbol: str, days_back: int = 7) -> list:
    """Fetch recent news articles for a stock."""
    if not NEWS_API_KEY:
        print("⚠️  No NewsAPI key found — using dummy sentiment")
        return []

    company_name = STOCK_NAMES.get(symbol, symbol.replace('.NS', ''))
    from_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={company_name}+stock+NSE&"
        f"from={from_date}&"
        f"language=en&"
        f"sortBy=relevancy&"
        f"apiKey={NEWS_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        articles = response.json().get('articles', [])
        return articles[:20]  # Top 20 articles
    except Exception as e:
        print(f"News fetch error: {e}")
        return []


def analyze_sentiment(symbol: str) -> dict:
    """Analyze sentiment from news for a stock."""
    articles = fetch_news(symbol)

    if not articles:
        return {
            'symbol': symbol,
            'sentiment_score': 0.0,
            'sentiment_label': 'NEUTRAL',
            'articles_analyzed': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'headlines': []
        }

    scores = []
    pos, neg, neu = 0, 0, 0
    headlines = []

    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '') or ''
        text = f"{title}. {description}"

        score = analyzer.polarity_scores(text)['compound']
        scores.append(score)
        headlines.append({'title': title, 'score': round(score, 3)})

        if score > 0.05:
            pos += 1
        elif score < -0.05:
            neg += 1
        else:
            neu += 1

    avg_score = sum(scores) / len(scores) if scores else 0

    label = 'VERY POSITIVE' if avg_score > 0.3 else \
            'POSITIVE' if avg_score > 0.05 else \
            'VERY NEGATIVE' if avg_score < -0.3 else \
            'NEGATIVE' if avg_score < -0.05 else 'NEUTRAL'

    return {
        'symbol': symbol,
        'sentiment_score': round(avg_score, 4),
        'sentiment_label': label,
        'articles_analyzed': len(articles),
        'positive_count': pos,
        'negative_count': neg,
        'neutral_count': neu,
        'headlines': headlines[:5]
    }


if __name__ == "__main__":
    symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']

    print("📰 News Sentiment Analysis\n")
    for sym in symbols:
        result = analyze_sentiment(sym)
        print(f"--- {sym} ---")
        print(f"  Score: {result['sentiment_score']} | {result['sentiment_label']}")
        print(f"  Articles: {result['articles_analyzed']} | +{result['positive_count']} -{result['negative_count']} ~{result['neutral_count']}")
        if result['headlines']:
            print("  Top headlines:")
            for h in result['headlines'][:3]:
                print(f"    [{h['score']:+.2f}] {h['title'][:80]}")
        print()