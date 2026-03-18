import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("air_quality.csv")

# Display first rows
print(data.head())

# Selecting features
X = data[['PM2.5','PM10','NO2','CO','SO2','O3']]
y = data['AQI']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train,y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
error = mean_absolute_error(y_test,predictions)
print("Mean Absolute Error:", error)

# Plot result
plt.scatter(y_test,predictions)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("AQI Prediction")
plt.show()
