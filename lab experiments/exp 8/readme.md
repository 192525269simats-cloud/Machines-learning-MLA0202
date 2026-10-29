
# Linear Regression Algorithm with Graph

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 7, 8, 9])

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Display results
print("Linear Regression Algorithm")
print("Slope:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))
print("Predicted values:", np.round(y_pred, 2))

# Predict for a new value
new_value = np.array([[9]])
prediction = model.predict(new_value)
print("Prediction for X = 9:", round(prediction[0], 2))

# Plot actual data and regression line
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Regression Line")

plt.title("Linear Regression")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.legend()
plt.grid(True)
plt.show()
