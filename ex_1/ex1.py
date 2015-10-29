__author__ = 'fabian'

import sklearn
import numpy as np
import vigra
import matplotlib.pyplot as plt
from sklearn.datasets import  load_digits
from sklearn import cross_validation

def dist_loop(X_train, X_test):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    dist_matrix = np.zeros((n_train, n_test))

    for i in xrange(n_train):
        for j in xrange(n_test):
            dist_matrix[i, j] = np.sqrt(np.sum(np.square(X_train[i] - X_test[j])))

    return dist_matrix

def dist_vec(X_train, X_test):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    train_tiled = np.tile(X_train, (n_test, 1, 1)).transpose((1, 0, 2))
    test_tiled = np.tile(X_test, (n_train, 1, 1))

    diff = train_tiled - test_tiled

    dist_matrix = np.sqrt(np.sum(np.square(diff), axis=2))

    return dist_matrix

digits = load_digits()

data = digits["data"]
images = digits["images"]
target = digits["target"]
target_names = digits["target_names"]

print data.dtype

img = np.reshape(data[99], (8, 8))

plt.figure()
plt.imshow(img, cmap="gray", interpolation=None)
plt.show()

X_all = data
Y_all = target

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(digits.data, digits.target, test_size=0.4, random_state=0)

dist_matrix_loop = dist_loop(X_train, X_test)
dist_matrix_vec = dist_vec(X_train, X_test)

