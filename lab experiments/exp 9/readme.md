
# Compare Linear and Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([2, 3, 5, 4, 6, 8, 9, 12])

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)

# Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)

# Display results
print("Linear Regression")
print("MSE:", round(mean_squared_error(y, linear_pred), 2))
print("R2 Score:", round(r2_score(y, linear_pred), 2))

print("\nPolynomial Regression")
print("MSE:", round(mean_squared_error(y, poly_pred), 2))
print("R2 Score:", round(r2_score(y, poly_pred), 2))

# Smooth curve for plotting
X_smooth = np.linspace(1, 8, 200).reshape(-1, 1)
linear_smooth = linear_model.predict(X_smooth)
poly_smooth = poly_model.predict(poly.transform(X_smooth))

# Plot comparison
plt.scatter(X, y, color="black", label="Actual Data")
plt.plot(X_smooth, linear_smooth, color="red",
         label="Linear Regression")
plt.plot(X_smooth, poly_smooth, color="blue",
         label="Polynomial Regression (Degree 2)")

plt.title("Linear vs Polynomial Regression")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.legend()
plt.grid(True)
plt.show()
