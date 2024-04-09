import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset from Excel file
data = pd.read_excel('house_sales_prices.xlsx')

# Extract features (X) and target variable (y)
X = data.drop('HousePrice', axis=1)  # Assuming 'HousePrice' is the column containing the target variable
y = data['HousePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict house prices on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Optionally, print the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Calculate correlation matrix
correlation_matrix = data.corr()

# Print correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)
