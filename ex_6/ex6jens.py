#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:AUTHOR: Jens Petersen
:ORGANIZATION: Heidelberg University Hospital; German Cancer Research Center
:CONTACT: jens.petersen@dkfz-heidelberg.de
:SINCE: Tue Dec 15 09:57:57 2015
:VERSION: 0.1

DESCRIPTION
-----------



REQUIRES
--------



TODO
----



"""

# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import pairwise
from scipy.sparse import linalg, csc_matrix
from scipy.spatial import cKDTree
import skimage.color as skc
import skimage.data as skd
import skimage.filters as skf
import time

# =============================================================================
# PROGRAM METADATA
# =============================================================================

__author__ = 'Jens Petersen'
__email__ = 'jens.petersen@dkfz-heidelberg.de'
__copyright__ = ''
__license__ = ''
__date__ = 'Tue Dec 15 09:57:57 2015'
__version__ = '0.1'

# =============================================================================
# CLASSES
# =============================================================================


class KernelRidgeRegression(object):

    def __init__(self, kernel="gaussian", jobs=1):

        self.jobs = jobs
        self.kernel = kernel
        self.cutoff = None
        self.sigma = None
        self.tau = None

        self._X = None
        self._y = None
        self._tree = None
        self._alpha = None

    def fit(self, X, y, sigma=1.0, tau=0.1, cutoff=0.001):

        self.cutoff = float(cutoff)
        self.sigma = float(sigma)
        self.tau = float(tau)

        self._X = X
        self._y = y
        self._tree = cKDTree(self._X)

        kernelMatrix = self.kernelMatrix(X, None, sigma, cutoff, self.jobs)
        inverseMatrix = csc_matrix(kernelMatrix + tau*np.identity(X.shape[0]))
        self._alpha = linalg.lsqr(inverseMatrix, y)[0]

        return self._alpha

    def predict(self, Xnew):

        g = self.kernelMatrix(Xnew, self._X, self.sigma, self.cutoff,
                              self.jobs)
        y = np.dot(g, self._alpha)

        return y

    def predictSingle(self, Xnew):

        y = np.empty(Xnew.shape[0])
#        maxDistance = - 2 * self.sigma**2 * np.log(
#            np.sqrt(2*np.pi) * self.sigma * self.cutoff)
        maxDistance = - 2 * self.sigma**2 * np.log(self.cutoff)

        for i in xrange(Xnew.shape[0]):
            print i
            indices = np.asarray(self._tree.query_ball_point(
                Xnew[i], maxDistance), dtype=np.int)
            g = self.kernelMatrix(Xnew[i], self._X[indices], self.sigma,
                                  self.cutoff)
            y[i] = np.dot(g, self._alpha[indices])

        return y

    def kernelMatrix(self, *args, **kwargs):

        if self.kernel == "gaussian":
            return self.gaussianKernel(*args, **kwargs)
        elif self.kernel == "quadratic":
            return self.quadraticKernel(*args, **kwargs)
        elif self.kernel == "polynomial":
            return self.polynomialKernel(*args, **kwargs)
        else:
            raise ValueError("Unknown kernel option")

    @staticmethod
    def gaussianKernel(X, X2=None, sigma=1.0, cutoff=None, jobs=1):

        sigma = float(sigma)

        if X2 is not None:
            distanceMatrix = pairwise.pairwise_distances(X, X2, n_jobs=jobs)
        else:
            distanceMatrix = pairwise.pairwise_distances(X, n_jobs=jobs)
#        result = 1 / np.sqrt(2*np.pi) / sigma *\
#            np.exp(-distanceMatrix / 2.0 / sigma**2)
        result = np.exp(-distanceMatrix / 2.0 / sigma**2)
        if cutoff is not None:
            result[result < cutoff] = 0

        return result

    @staticmethod
    def quadraticKernel():

        pass

    @staticmethod
    def polynomialKernel():

        pass

# =============================================================================
# METHODS
# =============================================================================


# =============================================================================
# MAIN METHOD
# =============================================================================


def main():

    sigma = 2.0
    tau = 0.1
    cutoff = 0.001
    factor = 10

    # load image
    original = skc.rgb2gray(plt.imread("nummernschild.jpg"))
    original = skf.gaussian_filter(original, 5.0)
    image = np.zeros(original.shape)

    # random downsampling
    indices = map(tuple, np.transpose(np.where(original > -1)))
    randomIndices = random.sample(indices, original.size/factor)
    for index in randomIndices:
        image[index] = original[index]
#    image = plt.imread("cc_90.png")

    coords = np.where(image != 0)
    X = np.transpose(coords)
    y = image[coords]

    coordsnew = np.where(image == 0)
    Xnew = np.transpose(coordsnew)
    result = np.zeros(image.shape)
    result[coords] = y

    regressor = KernelRidgeRegression(jobs=10)
    print "Start fitting"
    t0 = time.time()
    regressor.fit(X, y, sigma, tau, cutoff)
    print time.time() - t0

    print "Predicting single"
    t0 = time.time()
    ynew = regressor.predictSingle(Xnew)
    print time.time() - t0

    result[coordsnew] = ynew

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(original, cmap="gray")
    ax[1].imshow(image, cmap="gray")
    ax[2].imshow(result, cmap="gray")
    plt.show()

# =============================================================================
# RUN
# =============================================================================


if __name__ == "__main__":

    main()

