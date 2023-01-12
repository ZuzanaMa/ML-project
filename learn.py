import numpy as np
import os
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt


def load_data():
    data = []
    for path in os.listdir('data'):
        data += list(np.loadtxt('data\\' + path, delimiter=","))
    data = np.array(data)
    np.random.shuffle(data)

    return data[:, :-3], data[:, -3], data[:, -2], data[:, -1]

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


def learn(X, y):
    # TODO: filter outliers?
    # TODO: to normalize or to not normalize
    # vahovana lin reg?
    # sc_X = StandardScaler()
    # X = sc_X.fit_transform(X)
    print(np.shape(X))
    train = int(np.shape(X)[0]*0.8)
    val = int(np.shape(X)[0]*0.95)

    X_train = X[:train]
    X_val = X[train:val]
    y_train = y[:train]
    y_val = y[train:val]

    for i in range(1, 7):
        print(i)
        poly = PolynomialFeatures(degree = i)
        X_poly = poly.fit_transform(X_train)
        print(np.shape(X_poly))
  
        # poly.fit(X_poly, y)
        lin2 = LinearRegression()
        lin2.fit(X_poly, y_train)

        print(mean_squared_error(lin2.predict(poly.fit_transform(X_val)), y_val))

X, Yx, Yy, Yz = load_data()

from time import time
t = time()
# nenormalizujeme data - lin.reg
learn(X, Yx)
print(time()-t)

# 3 , normalizovane
# t = time()
# learn(X, Yy)
# print(time()-t)

# 2, nenormalizujeme 
# t = time()
# learn(X, Yz)
# print(time()-t)

force_vis()
