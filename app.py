import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

st.set_page_config(page_title="توقع سعر البيتكوين", layout="centered")
st.title("🔮 توقع اتجاه سعر البيتكوين + تحليل مشاعر الأخبار")

def get_news_sentiment(keyword="Bitcoin"):
    url = f"https://news.google.com/search?q={keyword}%20when%3A7d&hl=en-US&gl=US&ceid=US%3Aen"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        articles = soup.find_all('a', class_='DY5T1d', limit=10)
        sentiments = []
        for article in articles:
            text = article.get_text()
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
        return round(np.mean(sentiments), 3) if sentiments else 0.0
    except Exception:
        return 0.0

@st.cache_data
def load_data():
    df = yf.download("BTC-USD", start="2021-01-01", end="2024-12-31")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)

    # فلترة بيانات سعر الإغلاق وضبط الفهرس
    df = df[df['Close'].notnull()]
    df = df.reset_index(drop=True)

    # طباعة لفحص البيانات في سجل الأخطاء (لو عندك وصول للسيرفر)
    print("أول 10 قيم لسعر الإغلاق:")
    print(df['Close'].head(10))

    if df.empty:
        st.error("لا توجد بيانات كافية لتحليلها.")
        st.stop()

    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df.dropna(inplace=True)

    df['Tomorrow'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    return df.dropna()

df = load_data()

sentiment_score = get_news_sentiment("Bitcoin")
st.metric("📢 مشاعر الأخبار", f"{sentiment_score:.2f}")
st.line_chart(df['Close'])

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal']
X = df[features].copy()
X['Sentiment'] = sentiment_score
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

latest = X.iloc[[-1]]
pred = model.predict(latest)[0]
proba = model.predict_proba(latest)[0][pred]

if pred == 1:
    st.success(f"السعر مرشح للارتفاع 📈 (ثقة: {proba*100:.2f}%)")
else:
    st.error(f"السعر مرشح للانخفاض 📉 (ثقة: {proba*100:.2f}%)")

acc = accuracy_score(y_test, model.predict(X_test))
st.caption(f"دقة النموذج: {acc*100:.2f}%")
