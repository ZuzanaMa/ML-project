import numpy as np
import os
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

# load data from every file in data directory
def load_data():
    data = []
    for path in os.listdir('data'):
        data += list(np.loadtxt('data\\' + path, delimiter=","))
    data = np.array(data)
    np.random.shuffle(data)

    return data[:, :-3], data[:, -3], data[:, -2], data[:, -1]

# basic visualization
def force_vis():
    X, Yx, Yy, Yz = load_data()
    plt.plot(Yx, Yy, '.')
    plt.xlabel("Force X")
    plt.ylabel("Force Y")
    print(np.corrcoef(Yx, Yy))
    plt.show()
    plt.plot(Yy, Yz, '.')
    plt.xlabel("Force Y")
    plt.ylabel("Force Z")
    print(np.corrcoef(Yy, Yz))
    plt.show()
    plt.plot(Yx, Yz, '.')
    plt.xlabel("Force X")
    plt.ylabel("Force Z")
    print(np.corrcoef(Yx, Yz))
    plt.show()

# find degree of polynomial transformation
def learn(X, y):
    train = int(np.shape(X)[0]*0.8)
    val = int(np.shape(X)[0]*0.95)

    X_train = X[:train]
    X_val = X[train:val]
    y_train = y[:train]
    y_val = y[train:val]

    err_train = []
    err_val = []
    best_i = 1
    best_err = float('inf')

    for i in range(1, 8):
        poly = PolynomialFeatures(degree = i)
        X_poly = poly.fit_transform(X_train)
  
        lin2 = LinearRegression()
        lin2.fit(X_poly, y_train)

        err_train.append(mean_squared_error(lin2.predict(poly.fit_transform(X_train)), y_train))
        err_val.append(mean_squared_error(lin2.predict(poly.fit_transform(X_val)), y_val))
        
        if err_val[-1] < best_err:
            best_err = err_val[-1]
            best_i = i

    print('Best model si polynom -', best_i, 'degree')
    poly = PolynomialFeatures(degree = best_i)
    X_poly = poly.fit_transform(X_train)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y_train)

    print('test error:', mean_squared_error(lin2.predict(poly.fit_transform(X[val:])), y[val:]))

    plt.plot(list(range(1, 8)), err_train )
    plt.plot(list(range(1, 8)), err_val )
    plt.show()

# force_vis()
X, Yx, Yy, Yz = load_data()

print('====Force X====')
learn(X, Yx)
print('====Force Y====')
learn(X, Yy)
print('====Force Z====')
learn(X, Yz)


