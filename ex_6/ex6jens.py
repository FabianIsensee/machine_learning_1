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
import re
from sklearn.kernel_ridge import KernelRidge as kr
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

    distanceKernels = ("gaussian", "laplacian", "exponential")

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

    def fit(self, X, y, sigma=1.0, tau=0.1, cutoff=0.001, power=2, offset=1):

        self.cutoff = float(cutoff)
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.power = int(power)
        self.offset = float(offset)

        self._X = X
        self._y = y
        self._tree = cKDTree(self._X)

        kernelMatrix = self.kernelMatrix(X, None, sigma=sigma, cutoff=cutoff,
                                         jobs=self.jobs, power=power,
                                         offset=offset)
        inverseMatrix = csc_matrix(kernelMatrix + tau*np.identity(X.shape[0]))
        self._alpha = linalg.lsqr(inverseMatrix, y)[0]

        return self._alpha

    def predict(self, Xnew):

        g = self.kernelMatrix(Xnew, self._X, sigma=self.sigma,
                              cutoff=self.cutoff, jobs=self.jobs)
        y = np.dot(g, self._alpha)

        return y

    def predictSingle(self, Xnew):

        if self.kernel not in self.distanceKernels:
            return self.predict(Xnew)

        y = np.empty(Xnew.shape[0])
        maxDistance = self.getMaxDistance()
#        maxDistance = - 2 * self.sigma**2 * np.log(self.cutoff)

        for i in xrange(Xnew.shape[0]):
            print i
            indices = np.asarray(self._tree.query_ball_point(
                Xnew[i], maxDistance), dtype=np.int)
            g = self.kernelMatrix(Xnew[i], self._X[indices], sigma=self.sigma,
                                  cutoff=self.cutoff, power=self.power,
                                  offset=self.offset)
            y[i] = np.dot(g, self._alpha[indices])

        return y

    def kernelMatrix(self, X, X2=None, *args, **kwargs):

        if self.kernel == "gaussian":
            return self.gaussianKernel(X, X2, *args, **kwargs)
        elif self.kernel == "quadratic":
            return self.quadraticKernel(X, X2, *args, **kwargs)
        elif self.kernel == "polynomial":
            return self.polynomialKernel(X, X2, *args, **kwargs)
        elif self.kernel == "laplacian":
            return self.laplacianKernel(X, X2, *args, **kwargs)
        elif self.kernel == "exponential":
            return self.exponentialKernel(X, X2, *args, **kwargs)
        elif self.kernel == "rationalquadratic":
            return self.rationalQuadraticKernel(X, X2, *args, **kwargs)
        elif self.kernel == "multiquadric":
            return self.multiQuadricKernel(X, X2, *args, **kwargs)
        elif self.kernel == "cauchy":
            return self.cauchyKernel(X, X2, *args, **kwargs)
        else:
            raise ValueError("Unknown kernel option")

    def getMaxDistance(self):

        if self.kernel == "gaussian":
            return np.sqrt(-2 * self.sigma**2 * np.log(
                np.sqrt(2*np.pi) * self.sigma * self.cutoff))
        elif self.kernel == "laplacian":
            return -self.sigma * np.log(self.cutoff)
        elif self.kernel == "exponential":
            return -2 * self.sigma**2 * np.log(self.cutoff)
        else:
            raise ValueError("Not a distance kernel.")

    @staticmethod
    def gaussianKernel(X, X2=None, sigma=1.0, cutoff=None, jobs=1,
                       *args, **kwargs):

        sigma = float(sigma)

        if X2 is not None:
            distanceMatrix = pairwise.pairwise_distances(X, X2, n_jobs=jobs)
        else:
            distanceMatrix = pairwise.pairwise_distances(X, n_jobs=jobs)
        result = 1 / np.sqrt(2*np.pi) / sigma *\
            np.exp(-distanceMatrix**2 / 2.0 / sigma**2)
#        result = np.exp(-distanceMatrix / 2.0 / sigma**2)
        if cutoff is not None:
            result[result < cutoff] = 0

        return result

    @staticmethod
    def exponentialKernel(X, X2=None, sigma=1.0, cutoff=None, jobs=1,
                          *args, **kwargs):

        sigma = float(sigma)

        if X2 is not None:
            distanceMatrix = pairwise.pairwise_distances(X, X2, n_jobs=jobs)
        else:
            distanceMatrix = pairwise.pairwise_distances(X, n_jobs=jobs)
        result = np.exp(-distanceMatrix / 2.0 / sigma**2)
        if cutoff is not None:
            result[result < cutoff] = 0

        return result

    @staticmethod
    def quadraticKernel(X, X2=None, offset=1, *args, **kwargs):

        return KernelRidgeRegression.polynomialKernel(X, X2, power=2,
                                                      offset=offset)

    @staticmethod
    def polynomialKernel(X, X2=None, power=2, offset=1, *args, **kwargs):

        if X2 is not None:
            result = np.dot(X, X2.T)
        else:
            result = np.dot(X, X.T)

        return np.power(result + offset, power)

    @staticmethod
    def laplacianKernel(X, X2=None, sigma=1.0, cutoff=None, jobs=1,
                        *args, **kwargs):

        sigma = float(sigma)

        if X2 is not None:
            distanceMatrix = pairwise.pairwise_distances(X, X2, n_jobs=jobs)
        else:
            distanceMatrix = pairwise.pairwise_distances(X, n_jobs=jobs)
        result = np.exp(-distanceMatrix / sigma)
        if cutoff is not None:
            result[result < cutoff] = 0

        return result

    @staticmethod
    def rationalQuadraticKernel(X, X2=None, offset=1.0, jobs=1,
                                *args, **kwargs):

        assert (offset > 0)
        offset = float(offset)

        if X2 is not None:
            distanceMatrix = pairwise.pairwise_distances(X, X2, n_jobs=jobs)
        else:
            distanceMatrix = pairwise.pairwise_distances(X, n_jobs=jobs)
        result = 1 - distanceMatrix**2 / (distanceMatrix**2 + offset)

        return result

    @staticmethod
    def multiQuadricKernel(X, X2=None, offset=1.0, jobs=1, *args, **kwargs):

        offset = float(offset)
        if X2 is not None:
            distanceMatrix = pairwise.pairwise_distances(X, X2, n_jobs=jobs)
        else:
            distanceMatrix = pairwise.pairwise_distances(X, n_jobs=jobs)
        result = np.sqrt(distanceMatrix**2 + offset**2)

        return result

    @staticmethod
    def cauchyKernel(X, X2, sigma=1.0, jobs=1, *args, **kwargs):

        sigma = float(sigma)

        if X2 is not None:
            distanceMatrix = pairwise.pairwise_distances(X, X2, n_jobs=jobs)
        else:
            distanceMatrix = pairwise.pairwise_distances(X, n_jobs=jobs)
        result = 1 / (1 + distanceMatrix**2 / sigma**2)

        return result

# =============================================================================
# METHODS
# =============================================================================


# =============================================================================
# MAIN METHOD
# =============================================================================


def main():

    sigma = 3.0
    tau = 0.1
    cutoff = 0.001
    factor = 20
    offset = 0.2
    power = 1

    # load image
#    original = skc.rgb2gray(plt.imread("nummernschild.jpg"))
##    original = skf.gaussian_filter(original, 5.0)
#    image = np.zeros(original.shape)
#
#    # random downsampling
#    indices = map(tuple, np.transpose(np.where(original > -1)))
#    randomIndices = random.sample(indices, original.size/factor)
#    for index in randomIndices:
#        image[index] = original[index]
    image = plt.imread("cc_90.png")

    coords = np.where(image != 0)
    X = np.transpose(coords)
    y = image[coords]

    coordsnew = np.where(image == 0)
    Xnew = np.transpose(coordsnew)
    result = np.zeros(image.shape)
    result[coords] = y

    regressor = KernelRidgeRegression(kernel="cauchy", jobs=10)
    print "Start fitting"
    t0 = time.time()
    regressor.fit(X, y, sigma=sigma, tau=tau, cutoff=cutoff, offset=offset,
                  power=power)
    print time.time() - t0

    print "Predicting single"
    t0 = time.time()
    ynew = regressor.predictSingle(Xnew)
    print time.time() - t0

#    regressor = kr(kernel="sigmoid")
#    print "Fitting"
#    regressor.fit(X, y)
#    print "Predicting"
#    ynew = regressor.predict(Xnew)

    result[coordsnew] = ynew

    f, ax = plt.subplots(1, 2)
#    ax[0].imshow(original, cmap="gray")
    ax[0].imshow(image, cmap="gray")
    ax[1].imshow(result, cmap="gray")
    plt.show()

# =============================================================================
# RUN
# =============================================================================


if __name__ == "__main__":

    main()

