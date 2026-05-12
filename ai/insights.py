# ai/insights.py
import requests
import os
from dotenv import load_dotenv

try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def generate_insights(symbol, technical_signals, sentiment, prediction=None):
    """Generate insights using Groq (free, works on cloud)."""

    prompt = f"""
You are a senior Indian stock market analyst. Analyze the following data for {symbol} and give a clear, concise investment insight in 4-5 sentences. Mention key risks. Be direct, not generic.

TECHNICAL SIGNALS:
- Current Price: ₹{technical_signals.get('current_price')}
- RSI: {technical_signals.get('RSI')} ({technical_signals.get('RSI_signal')})
- MACD Direction: {technical_signals.get('MACD_direction')}
- Bollinger Band Signal: {technical_signals.get('BB_signal')}
- Price vs EMA20: {technical_signals.get('price_vs_EMA20')}
- Price vs EMA50: {technical_signals.get('price_vs_EMA50')}
- Overall Technical Score: {technical_signals.get('overall_score')}/5
- Overall Signal: {technical_signals.get('overall_signal')}

NEWS SENTIMENT:
- Sentiment Score: {sentiment.get('sentiment_score')} ({sentiment.get('sentiment_label')})
- Articles Analyzed: {sentiment.get('articles_analyzed')}
- Positive: {sentiment.get('positive_count')} | Negative: {sentiment.get('negative_count')}

{"LSTM PREDICTION: " + str(prediction) if prediction else ""}

Give your analysis as:
1. VERDICT: (BUY / SELL / HOLD)
2. REASONING: (2-3 sentences)
3. KEY RISK: (1 sentence)
4. CONFIDENCE: (Low / Medium / High)
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",  # Free Llama3 on Groq
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            },
            timeout=30
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"AI insights unavailable: {e}"