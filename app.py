import streamlit as st
import pandas as pd

st.title("📊 Sales Forecast Dashboard")

df = pd.read_csv("forecast.csv")

st.dataframe(df)
st.line_chart(df.set_index("date")["sales"])