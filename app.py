import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

st.title("تحليل البيتكوين مع مؤشرات RSI و MACD")

@st.cache_data
def load_data(ticker="BTC-USD", period="6mo"):
    df = yf.download(ticker, period=period)
    df.dropna(inplace=True)
    return df

df = load_data()

# حساب RSI
rsi = RSIIndicator(close=df['Close'], window=14).rsi()
df['RSI'] = rsi

# حساب MACD
macd_indicator = MACD(close=df['Close'])
df['MACD'] = macd_indicator.macd()
df['MACD_signal'] = macd_indicator.macd_signal()

st.subheader("البيانات التاريخية للبيتكوين")
st.dataframe(df.tail(10))

st.subheader("مخطط سعر الإغلاق مع مؤشرات RSI و MACD")

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

ax1.plot(df.index, df['Close'], label='سعر الإغلاق')
ax1.set_ylabel('السعر')
ax1.legend()

ax2.plot(df.index, df['RSI'], label='RSI', color='orange')
ax2.axhline(70, color='red', linestyle='--')
ax2.axhline(30, color='green', linestyle='--')
ax2.set_ylabel('RSI')
ax2.legend()

ax3.plot(df.index, df['MACD'], label='MACD', color='purple')
ax3.plot(df.index, df['MACD_signal'], label='MACD Signal', color='green')
ax3.set_ylabel('MACD')
ax3.legend()

st.pyplot(fig)
