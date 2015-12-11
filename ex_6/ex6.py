__author__ = 'fabian'
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as pwd
from scipy.sparse import dok_matrix, linalg, csc_matrix, coo_matrix, csr_matrix
import IPython


class KernelRidgeRegression(object):
    def __init__(self):
        self._alpha = None
        self._x = None
        self._y = None
        self._sigma = None

    def compute_alpha(self, x, y, tau, sigma = 1, cutoff = 0.001):
        self._x = x
        self._y = y
        self._sigma = sigma
        n_instances = x.shape[0]
        dist_matrix = pwd.pairwise_distances(x, n_jobs=16)
        kernel_matrix = self.gaussian(dist_matrix, sigma)
        kernel_matrix[kernel_matrix < cutoff] = 0
        kernel_matrix += float(tau) * np.identity(n_instances)
        kernel_matrix = csc_matrix(kernel_matrix)

        res = linalg.lsqr(kernel_matrix, y)
        self._alpha = res[0]

    @staticmethod
    def gaussian(matrix, sigma):
        return np.exp(-(matrix)/(2.*sigma**2.))

    def compute_g(self, x_new, sigma):
        return self.gaussian(pwd.pairwise_distances(x_new, self._x, n_jobs=16), sigma)

    def fit(self, x, y, tau, sigma = 1.0, cutoff = 0.001):
        self.compute_alpha(x, y, tau, sigma, cutoff)

    def predict(self, new_x):
        assert(self._alpha is not None, "train first")
        g = self.compute_g(new_x, self._sigma)
        y_new = np.dot(g, self._alpha)
        return y_new

class KernelRegression(object):
    def __init__(self):
        self._x = None
        self._y = None
        self._sigma = None

    def fit(self, x, y, sigma):
        self._x = x
        self._y = y
        self._sigma = sigma

    def predict(self, x_new):
        pairwise_dist = pwd.pairwise_distances(x_new, self._x, n_jobs=16)
        g = KernelRidgeRegression.gaussian(pairwise_dist, sigma=self._sigma)
        sum_per_row = np.sum(g, axis=1)
        y_new = np.dot(g, self._y)
        return y_new / sum_per_row


if __name__ == "__main__":
    img = plt.imread("cc_90.png")
    coords = np.where(img != 0)
    x = np.array([coords[0], coords[1]]).transpose()
    y = img[coords]

    sigma = [1., 1.5, 2.5, 3.5]
    tau = [0.1, 0.2, 0.3, 0.4, 0.5, 1., 2.]
    for this_sigma in sigma:
        for this_tau in tau:
            print this_sigma, this_tau
            krr = KernelRidgeRegression()
            krr.fit(x, y, this_tau, this_sigma, 0.001)

            new_coords = np.where(img == 0)
            x_new = np.array([new_coords[0], new_coords[1]]).transpose()
            y_new = krr.predict(x_new)

            im_krr = np.array(img)
            im_krr[new_coords] = y_new
            plt.imsave("krr_result_sigma_%1.2f_tau_%1.2f.png" % (this_sigma, this_tau), im_krr, cmap="gray")
        kr = KernelRegression()
        kr.fit(x, y, this_sigma)
        y_new = kr.predict(x_new)
        im_kr = np.array(img)
        im_kr[new_coords] = y_new
        plt.imsave("kr_result_sigma_%1.2f.png" % this_sigma, im_kr, cmap="gray")