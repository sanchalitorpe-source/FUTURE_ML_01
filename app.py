import streamlit as st
import pandas as pd
import os

st.title("📊 Sales Forecast Dashboard")

# ✅ Replace this block with the line below
csv_path = os.path.join(os.path.dirname(__file__), "forecast.csv")

if not os.path.exists(csv_path):
    df = pd.DataFrame({
        "date": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
        "sales": [120, 135, 128, 145, 160, 172]
    })
else:
    df = pd.read_csv(csv_path)

st.dataframe(df)
st.line_chart(df.set_index("date")["sales"])