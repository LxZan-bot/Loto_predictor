import streamlit as st
import pandas as pd
from model import train_model
from predictor import predict_next

st.title("🔮 Loto 6 ခန့်မှန်းမှု")

df = pd.read_csv('loto_data.csv')
model = train_model()
last_draw = df.iloc[-1, 1:7].tolist()

st.write("နောက်ဆုံးထွက်ခဲ့တဲ့ဂဏန်းများ:", last_draw)
if st.button("ခန့်မှန်းမယ်"):
    result = predict_next(model, last_draw)
    st.success(f"🎯 ခန့်မှန်းထားသောဂဏန်းများ: {result}")
