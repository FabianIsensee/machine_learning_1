#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:AUTHOR: Jens Petersen
:ORGANIZATION: Heidelberg University Hospital; German Cancer Research Center
:CONTACT: jens.petersen@dkfz-heidelberg.de
:SINCE: Wed Dec 16 16:42:55 2015
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

import IPython
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# =============================================================================
# PROGRAM METADATA
# =============================================================================

__author__ = 'Jens Petersen'
__email__ = 'jens.petersen@dkfz-heidelberg.de'
__copyright__ = ''
__license__ = ''
__date__ = 'Wed Dec 16 16:42:55 2015'
__version__ = '0.1'

# =============================================================================
# CLASSES
# =============================================================================


# =============================================================================
# METHODS
# =============================================================================

def fitCircle(data, mode="algebraic"):

    data = np.asarray(data)

    # exact fit for three points
    if data.shape[0] == 3:

        b = data[:, 0]**2 + data[:, 1]**2
        A = np.append(data, np.ones((data.shape[0], 1)), axis=1)
        x = np.dot(np.linalg.inv(A), b)

        x_m, y_m = x[0]/2., x[1]/2.
        r = np.sqrt(x[2] + x_m**2 + y_m**2)

        return x_m, y_m, r

    else:

        if mode == "algebraic":

            b = data[:, 0]**2 + data[:, 1]**2
            A = np.append(data, np.ones((data.shape[0], 1)), axis=1)
            x = np.linalg.lstsq(A, b)[0]

            x_m, y_m = x[0]/2., x[1]/2.
            r = np.sqrt(x[2] + x_m**2 + y_m**2)

            return x_m, y_m, r

        elif mode == "lm":

            sol = optimize.root(distanceToCircleOpt, [1, 1, 1],
                                args=data, method="lm", jac=True)
            return sol.x

        else:

            raise ValueError("Unknown fit mode.")


def distanceToCircle(center, radius, point):

    return np.abs(radius - np.sqrt(
        (point[0] - center[0])**2 + (point[1] - center[1])**2))


def distanceToCircleOpt(beta, x):

    result = np.zeros(len(x))
    jacobian = np.zeros((len(x), len(beta)))

    for i, point in enumerate(x):
        root = np.sqrt((beta[0] - point[0])**2 + (beta[1] - point[1])**2)
        f = root - beta[2]
        df0 = (beta[0] - point[0]) / root
        df1 = (beta[1] - point[1]) / root
        df2 = -1
        result[i] = f
        jacobian[i, :] = np.asarray([df0, df1, df2])

    return result, jacobian


def arrayDifference(arr1, arr2):

    set1 = set(map(tuple, arr1.tolist()))
    set2 = set(map(tuple, arr2.tolist()))
    return np.asarray(list(set1 - set2))


def RANSAC(data, epsilon, numberOfTrials):

    # initialize number of inliers
    bestNumberOfInliers = 0
    bestCenter = None
    bestRadius = 0
    bestInliers = None

    for _ in xrange(numberOfTrials):

        # select three points at random
        indices = np.random.choice(np.arange(data.shape[0]), 3, replace=False)
        points = data[indices]

        # determine circle passing all points
        x_m, y_m, radius = fitCircle(points)
        center = (x_m, y_m)

        # count inliers
        inliers = []
        for datapoint in data:
            if distanceToCircle(center, radius, datapoint) < epsilon:
                inliers.append(datapoint)
        numberOfInliers = len(inliers)

        if numberOfInliers > bestNumberOfInliers:
            bestNumberOfInliers = numberOfInliers
            bestCenter = center
            bestRadius = radius
            bestInliers = inliers

    return bestCenter, bestRadius, np.asarray(bestInliers)


# =============================================================================
# MAIN METHOD
# =============================================================================


def main():

    # load data
    data = np.load("circles.npy")
    epsilon = 0.05
    repetitions = 1
    trials = 100
    linewidth = 2

    # initialize plot
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    size = 5000 * (np.max(data) - np.min(data)) / float(data.shape[0])
    limits = [-0.2, 1.2]
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")

    # plot data
    ax.plot(data[:, 0], data[:, 1], 'bo', markeredgewidth=0, markersize=size,
            alpha=0.5)

    # fit and plot fits
    for _ in xrange(repetitions):

        center, radius, inliers = RANSAC(data, epsilon, trials)

        # for 1.4, select subset of inliers and add outliers
        numberOfInliers = len(inliers)
        randomIndices = np.random.choice(np.arange(numberOfInliers),
                                         numberOfInliers/10)
        randomSubset = inliers[randomIndices]
        randomIndices = np.random.choice(np.arange(data.shape[0]), 20)
        randomOutliers = data[randomIndices]
        randomSubset = np.append(randomSubset, randomOutliers, axis=0)

#        ax.plot(inliers[:, 0], inliers[:, 1], 'ro', markeredgewidth=0,
#                markersize=size)
#        ransacCircle = plt.Circle(center, radius=radius, fill=False,
#                                  linewidth=linewidth)
#        ax.add_patch(ransacCircle)
#
#        x_m, y_m, r = fitCircle(inliers, mode="lm")
#        fittedCircle = plt.Circle((x_m, y_m), radius=r, fill=False,
#                                  color="green", linewidth=linewidth)
#        ax.add_patch(fittedCircle)

        ax.plot(randomSubset[:, 0], randomSubset[:, 1], 'ro',
                markeredgewidth=0, markersize=size)
        x_m, y_m, r = fitCircle(randomSubset, mode="algebraic")
        algebraicCircle = plt.Circle((x_m, y_m), radius=r, fill=False,
                                     color="green", linewidth=linewidth)
        ax.add_patch(algebraicCircle)
        x_m, y_m, r = fitCircle(randomSubset, mode="lm")
        lmCircle = plt.Circle((x_m, y_m), radius=r, fill=False,
                              linewidth=linewidth)
        ax.add_patch(lmCircle)

        data = arrayDifference(data, inliers)

    plt.show()

# =============================================================================
# RUN
# =============================================================================


if __name__ == "__main__":

    main()

