from sklearn.ensemble import RandomForestRegressor
import numpy as np

X = np.linspace(-100, 100, 100)
y1 = hyperbola(X)
y2 = polynom_3(X)
regr = RandomForestRegressor(max_depth=6, random_state=0)
regr.fit(X, y1)
y_pred1 = regr.predict(X)
mse1 = (y1 - y_pred1) ** 2 / 100

regr.fit(X, y2)
y_pred2 = regr.predict(X)
mse2 = (y2 - y_pred2) ** 2 / 100

print("MSE ", mse1, " ", mse2)
