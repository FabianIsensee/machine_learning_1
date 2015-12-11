__author__ = 'fabian'

import numpy as np
import IPython
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import lsqr

def makeA(shape, alphas, num_sensor_pixels=None):
    if num_sensor_pixels is None:
        num_sensor_pixels = int(np.ceil(np.sqrt(shape[0]**2 + shape[1]**2)))
        if num_sensor_pixels % 2 == 0:
            num_sensor_pixels += 1

    shape_of_a = (shape[0] * shape[1], len(alphas) * num_sensor_pixels)
    a = np.zeros(shape_of_a)
    center_coordinate = ((shape[0] - 1.) / 2., (shape[1] - 1.) / 2.)
    for al, alpha in enumerate(alphas):
        # IPython.embed()
        sensor_vec = np.array([- 1., np.tan((-alpha)/180.*np.pi)])
        sensor_vec /= np.linalg.norm(sensor_vec)
        if alpha == 90:
            sensor_vec = np.array([-1, 0])
        if alpha == 0:
            sensor_vec = np.array([0, -1])
        first_center_coordinate_sensor = center_coordinate - ((num_sensor_pixels - 1.) / 2.) * sensor_vec
        offset = al * num_sensor_pixels
        for y in xrange(shape[1]):
            for x in xrange(shape[0]):
                tmp = np.array([x, y]) - first_center_coordinate_sensor
                proj = np.dot(tmp, sensor_vec)
                lower_sensor_pixel = np.floor(proj)
                upper_sensor_pixel = np.ceil(proj)
                proportion_lower = 1. - proj % 1
                a[x + y * shape[0], offset + lower_sensor_pixel] += proportion_lower
                if upper_sensor_pixel != lower_sensor_pixel:
                    proportion_upper = 1. - proportion_lower
                    a[x + y * shape[0], offset + upper_sensor_pixel] += proportion_upper
    return a

def makeA_jens(shape, alphas, num_sensor_pixels=None):

    if num_sensor_pixels is None:
        num_sensor_pixels = int(np.ceil(np.sqrt(shape[0]**2 + shape[1]**2)))
        if num_sensor_pixels % 2 == 0:
            num_sensor_pixels += 1

    A = np.zeros((shape[0] * shape[1], len(alphas) * num_sensor_pixels))

    for al, alpha in enumerate(alphas):

        alpha_rad = alpha / 180. * np.pi
        offset = al * num_sensor_pixels

        for x in xrange(shape[0]):
            for y in xrange(shape[1]):

                proj = np.cos(alpha_rad) * (x - 0.5*(shape[0]-1) - 0.5*(num_sensor_pixels-1)*(0-np.cos(alpha_rad)))\
                       - np.sin(alpha_rad) * (y - 0.5*(shape[1]-1) - 0.5*(num_sensor_pixels-1)*(0+np.sin(alpha_rad)))
                lower_sensor_pixel = np.floor(proj)
                upper_sensor_pixel = np.ceil(proj)
                proportion_lower = 1. - proj % 1
                A[x + y * shape[0], offset + lower_sensor_pixel] += proportion_lower
                if upper_sensor_pixel != lower_sensor_pixel:
                    proportion_upper = 1. - proportion_lower
                    A[x + y * shape[0], offset + upper_sensor_pixel] += proportion_upper

    return A





if __name__ == "__main__":
    image_y = np.load("hs_tomography_2/y_77_.npy")
    image_alphas = np.load("hs_tomography_2/y_77_alphas.npy").astype("float")

    image_flattened = image_y.flatten()
    c = np.array([-77,-33,-12, -3,21,42,50,86]).astype("float")

    a = makeA_jens((77,77), image_alphas).transpose()

    import matplotlib.pyplot as plt
    plt.imshow(a.transpose(), cmap="gray", interpolation="none")
    plt.close()
    # plt.show()
    a_sparse = dok_matrix(a)
    res = lsqr(a_sparse, image_y)
    res_new = res.reshape((77, 77))
    plt.imshow(res_new, cmap="gray")
    plt.show()
    IPython.embed()

