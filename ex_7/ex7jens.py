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

import matplotlib.pyplot as plt
import numpy as np

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

def fitCircle(points):

    b = []
    A = []
    for point in points:
        b.append(-point[0]**2-point[1]**2)
        A.append([1, point[0], point[1]])
    b = np.asarray(b)
    A = np.asarray(A)
    x = np.dot(np.linalg.inv(A), b)

    x_m = x[1]/2.
    y_m = x[2]/2.
    r = np.sqrt(x_m**2 + y_m**2 - x[0])

    return x_m, y_m, r


def distanceToCircle(center, radius, point):

    return np.abs(radius - np.sqrt(
        (point[0] - center[0])**2 + (point[1] - center[1])**2))


def RANSAC(data, epsilon):

    # initialize number of inliers
    numberOfInliers = 0
    lastNumberOfInliers = 0

    while (numberOfInliers <= lastNumberOfInliers or lastNumberOfInliers == 0):

        lastNumberOfInliers = numberOfInliers

        # select three points at random
        indices = np.random.choice(np.arange(data.shape[0]), 3, replace=False)
        points = []
        for index in indices:
            points.append(data[index])

        # determine circle passing all points
        x_m, y_m, radius = fitCircle(points)
        center = (x_m, y_m)

        # count inliers
        inliers = []
        for datapoint in data:
            if distanceToCircle(center, radius, datapoint) < epsilon:
                inliers.append(datapoint)
        numberOfInliers = len(inliers)

    return center, radius, inliers

# =============================================================================
# MAIN METHOD
# =============================================================================


def main():

#    # load data
#    data = np.load("circles.npy")
#
#    # initialize plot
#    f = plt.figure()
#    ax = f.add_subplot(1, 1, 1)
#    r_point = (np.max(data) - np.min(data)) / float(data.shape[0])
#
#    # plot data
#    for coordinates in data:
#        circle = plt.Circle(coordinates, radius=r_point, fill=False)
#        ax.add_patch(circle)
#
#    # fit and plot fits
#    center, radius, _ = RANSAC(data, 0.01)
#    print center, radius
#    circle = plt.Circle(center, radius=radius, fill=False)
#
#    plt.show()

    randomPoints = np.random.rand(3, 2)
    x_m, y_m, r = fitCircle(randomPoints)

    #test
    print r
    for point in randomPoints:
        print np.sqrt((point[0] - x_m)**2 + (point[1] - y_m)**2)

    print randomPoints
    print x_m, y_m, r

    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.scatter(randomPoints[:, 0], randomPoints[:, 1])
    circle = plt.Circle((x_m, y_m), radius=r)
    ax.add_patch(circle)
    plt.show()

# =============================================================================
# RUN
# =============================================================================


if __name__ == "__main__":

    main()

