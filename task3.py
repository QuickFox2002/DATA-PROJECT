from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
california_housing = fetch_california_housing()

# Convert to pandas DataFrame
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['Price'] = california_housing.target

# Data Preprocessing
df.fillna(df.mean(), inplace=True)  # Handle missing values
df = pd.get_dummies(df)  # Convert categorical variables to numeric

# Normalize numerical features
scaler = StandardScaler()
X = df.drop('Price', axis=1)
y = df['Price']
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate Models
y_pred_lr = lr_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

print("Linear Regression Model:")
print(f"R² Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lr):.4f}")

print("\nDecision Tree Model:")
print(f"R² Score: {r2_score(y_test, y_pred_dt):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_dt):.4f}")

# User Input for Prediction
rooms = float(input("Enter number of rooms: "))
location = float(input("Enter location value: "))
size = float(input("Enter size of the house: "))

# Feature vector for prediction
input_data = np.array([[rooms, location, size]])

# Standardize the input
input_data_scaled = scaler.transform(input_data)

# Predict with the best model (Linear Regression)
house_price = lr_model.predict(input_data_scaled)
print(f"The predicted house price is: ${house_price[0]:,.2f}")
