import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

st.title("تطبيق تحليل بيتكوين باستخدام الذكاء الاصطناعي")

@st.cache_data
def load_data():
    # تحميل البيانات من Yahoo Finance
    df = yf.download("BTC-USD", start="2021-01-01", end="2024-12-31")
    
    # اختيار الأعمدة المهمة
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # تنظيف البيانات: حذف القيم الفارغة
    df.dropna(inplace=True)
    
    # حذف الصفوف التي تحتوي على قيم مفقودة في 'Close'
    df = df[df['Close'].notnull()]
    df.reset_index(drop=True, inplace=True)
    
    # حساب مؤشر RSI
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()
    
    # حساب مؤشر MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # حذف الصفوف التي تحتوي على قيم مفقودة بعد حساب المؤشرات
    df.dropna(inplace=True)
    
    # الهدف: هل سعر الغد أعلى من سعر اليوم (تصنيف ثنائي)
    df['Tomorrow'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    return df

df = load_data()

st.write("معاينة البيانات بعد التنظيف والحساب:")
st.dataframe(df.head())

# يمكنك هنا إضافة المزيد من التحليل أو النمذجة

st.write("مؤشرات RSI و MACD تم حسابها بنجاح.")
