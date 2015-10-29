__author__ = 'fabian'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  load_digits
from sklearn import cross_validation
import os
from scipy.stats import mode

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

def show_image(data):
    n_instances = data.shape[0]
    index = np.round(np.random.random() * (n_instances-1))
    img = np.reshape(data[index], (8, 8))

    plt.figure()
    plt.imshow(img, cmap="gray", interpolation=None)
    plt.show()

def predict_test(dist_matrix, Y_train):
    # dist matrix is (n_train by n_test)
    return Y_train[np.argmin(dist_matrix, axis=0)]

def predict_test_kNN(dist_matrix, Y_train, k=5):
    sorted_by_nearest = np.argsort(dist_matrix, axis=0)
    class_labels = Y_train[sorted_by_nearest[:k, :]]
    return mode(class_labels)[0][0]



if __name__ == "__main__":
    digits = load_digits()

    data = digits["data"]
    images = digits["images"]
    target = digits["target"]
    target_names = digits["target_names"]

    print data.dtype
    show_image(data)

    X_all = data
    Y_all = target

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(digits.data, digits.target, test_size=0.4, random_state=0)

    start_loop_time = os.times()[4]
    dist_matrix_loop = dist_loop(X_train, X_test)
    stop_loop_time = os.times()[4]

    start_vec_time = os.times()[4]
    dist_matrix_vec = dist_vec(X_train, X_test)
    stop_vec_time = os.times()[4]

    print("loop: %f\tvec:%f" % (stop_loop_time - start_loop_time, stop_vec_time - start_vec_time))

    # restrict data to 1's and 3's
    number_1 = 1
    number_2 = 3

    data = digits.data[(digits.target == number_2) | (digits.target == number_1)]
    target = digits.target[(digits.target == number_2) | (digits.target == number_1)]

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data, target, test_size=0.4, random_state=0)
    dist_matrix = dist_vec(X_train, X_test)
    Y_pred = predict_test(dist_matrix, Y_train)

    print "Correct classification rate of number '%i': %02.2f%%" %(number_1, np.sum(Y_pred[Y_test==number_1]==Y_test[Y_test==number_1]).astype("float")/np.sum(Y_test==number_1)*100)

    for k in [1, 3, 5, 9, 17, 33, 55]:
        Y_pred = predict_test_kNN(dist_matrix, Y_train, k)
        print "Correct classification k=%i rate of number '%i': %02.2f%%" %(k, number_1, np.sum(Y_pred[Y_test==number_1]==Y_test[Y_test==number_1]).astype("float")/np.sum(Y_test==number_1)*100)

    print "\nCross-Validation:"
    for n in [2, 5, 10]:
        kf = cross_validation.KFold(digits.data.shape[0], n, shuffle=True)
        errors = []
        for k, indices in enumerate(kf):
            X_train = digits.data[indices[0]]
            X_test = digits.data[indices[1]]
            Y_train = digits.target[indices[0]]
            Y_test = digits.target[indices[1]]

            dist_matrix = dist_vec(X_train, X_test)
            Y_pred = predict_test(dist_matrix, Y_train)
            errors.append(100. - np.sum(Y_pred==Y_test).astype("float")/len(Y_test)*100.)
        print "N=%i, mean err (%%) = %02.2f" % (n, np.mean(errors))
