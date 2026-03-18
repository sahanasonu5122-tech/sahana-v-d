import streamlit as st
import pandas as pd
import numpy as np
import mathplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import linearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.title("PragyanAI Taxi fare prediction App(end-to-end ML)")
@st.cache_data
def load_data():
  url = "taxis.csv"
  df = pd.read_csv(url)
  df = df.convent_dtypes()
  st.write(df.head())
  return df
  df = load_data()
  st.subheader("PragyanAI dataset preview")

df = df[['distance', 'fare']].dropna()
df['distance'] = pd.to_numeric(df['distance'], error='coerce')
df['fare'] = pd.to_numeric(df['fare'], errors='coerce')

x = df[['distance']]
y = df['fare']

x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("  📉 Model performance")
st.write(f"R² score: {r2:.2f}")
st.write(f"RMSE:.2f}")
st.subheader("🧱enter trio details")
distance = st.number_input(
  "step 1: enter distance(km)",
  min_value=0.0,
  value=5.0
)
passengers = st.number_input(
  "step 2: number of passengers",
  min_value=1,
  value=1
)
hour = st.number_input(
  "step 3: hour of day(0-23)",
  min_value=0,
  max_value=23,
  value=12
)
if st.button("predict fare"):
  input_data = np.array([[distance]])
  predictaion = model.predict(input_data)
  st.success(f"💰 estimated fare: ${prediction[0]:.2f}")
