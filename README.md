# -Prediction-using-Linear-Regression
House Price Prediction using Linear Regression
House Price Prediction using Linear Regression
================================================

#This project demonstrates a simple implementation of linear regression using Python's
scikit-learn library. It reads housing data (area vs. price), trains a model, predicts prices for new areas,
and visualizes the results. It is intended for educational and reference purposes.



# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load training data
df = pd.read_csv(r"C:\\Users\\muham\\AppData\\Roaming\\Python\\Book11.csv")

# Plot the training data
%matplotlib inline
plt.figure(figsize=(8,6))
plt.xlabel('Area (square feet)')
plt.ylabel('Price (US$)')
plt.title('House Price Prediction')
plt.scatter(df.area, df.price, color='blue', marker='*', label='Actual Data')

# Train linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# Predict on a single value (example)
predicted_price = reg.predict([[3300]])
print(f"Predicted price for 3300 sq ft: ${predicted_price[0]:,.2f}")
print(f"Slope (coefficient): {reg.coef_[0]:.2f}")
print(f"Intercept: {reg.intercept_:.2f}")

# Load new areas to predict
d = pd.read_csv(r"C:\\Users\\muham\\AppData\\Roaming\\Python\\areas.csv")

# Predict prices
p = reg.predict(d)
d['prices'] = p

# Save predictions
d.to_csv("Prediction.csv", index=False)
print("\nPredictions saved to 'Prediction.csv'")

# Plot regression line
plt.plot(df.area, reg.predict(df[['area']]), color='green', label='Regression Line')
plt.legend()
plt.grid(True)
plt.show()
 
