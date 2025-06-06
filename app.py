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

st.set_page_config(page_title="توقع سعر البيتكوين مع تحليل المشاعر", layout="centered")
st.title("🔮 توقع اتجاه سعر البيتكوين مع تحليل مشاعر الأخبار")

# دالة لجلب وتحليل مشاعر الأخبار
def get_news_sentiment(keyword="Bitcoin"):
    url = f"https://news.google.com/search?q={keyword}%20when%3A7d&hl=en-US&gl=US&ceid=US%3Aen"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        articles = soup.find_all('a', attrs={'class': 'DY5T1d'}, limit=10)
        sentiments = []
        for article in articles:
            text = article.get_text()
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
        if sentiments:
            return round(sum(sentiments) / len(sentiments), 3)
    except Exception as e:
        st.error(f"خطأ في جلب الأخبار: {e}")
    return 0.0

@st.cache_data
def load_data():
    df = yf.download("BTC-USD", start="2021-01-01", end="2024-12-31")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # تنظيف البيانات أولاً
    df.dropna(inplace=True)

    # التأكد من أن البيانات مش فاضية
    if df.empty or df['Close'].isnull().all().item():
    # المؤشرات الفنية
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # حذف الصفوف اللي فيها NaN بعد المؤشرات
    df.dropna(inplace=True)

    df['Tomorrow'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    return df.dropna()


df = load_data()
st.subheader("📊 حركة السعر التاريخية")
st.line_chart(df['Close'])

# جلب درجة المشاعر
sentiment_score = get_news_sentiment("Bitcoin")
st.markdown(f"### 📢 درجة المشاعر من الأخبار: {sentiment_score}")

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal']

X = df[features][:-1]
y = df['Target'][:-1]

# إضافة عمود المشاعر للبيانات (مكرر لنفس القيمة لأن المشاعر حالياً من آخر أخبار)
X['Sentiment'] = sentiment_score

features.append('Sentiment')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

latest_data = X.iloc[-1:]
prediction = model.predict(latest_data)[0]
probability = model.predict_proba(latest_data)[0][prediction]

st.subheader("📈 توقع الاتجاه القادم:")
if prediction == 1:
    st.success(f"يتوقع أن السعر سيرتفع 📈 (نسبة الثقة: {probability*100:.2f}%)")
else:
    st.error(f"يتوقع أن السعر سينخفض 📉 (نسبة الثقة: {probability*100:.2f}%)")

accuracy = accuracy_score(y_test, model.predict(X_test))
st.markdown(f"✅ **دقة النموذج:** {accuracy*100:.2f}%")
