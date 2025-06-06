import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import MACD

st.title("تطبيق تحليل بيتكوين باستخدام الذكاء الاصطناعي")

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data
def load_data():
    df = yf.download("BTC-USD", start="2021-01-01", end="2024-12-31")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    df = df[df['Close'].notnull()]
    df.reset_index(drop=True, inplace=True)
    
    df['RSI'] = compute_rsi(df['Close'], 14)
    
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    df.dropna(inplace=True)
    
    df['Tomorrow'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    return df

df = load_data()

st.write("معاينة البيانات بعد التنظيف والحساب:")
st.dataframe(df.head())

st.write("مؤشرات RSI و MACD تم حسابها بنجاح.")
