__author__ = "Fabian"
import sys
sys.path.append("../ex_2")
import numpy as np
import h5py
import feature_selection

class NaiveBayes(object):
    def __init__(self):
        self._X = None
        self._Y = None
        self._histograms = None
        self._priors = None
        self._n_features = None
        self.__classes = None

    def train(self, x, y, n_bins = None):
        self._X = x
        self._Y = y

        self.__classes = np.unique(self._Y)

        n_instances, self._n_features = self._X.shape

        self._histograms = []
        self._priors = []

        if n_bins is not None:
            n_bins_all_features = np.ones(self._n_features) * n_bins
        else:
            raise NotImplementedError("automatic determination of number of features missing")

        for current_class in self.__classes:
            self._priors = np.sum(self._Y == current_class) / float(n_instances)

            histograms_per_class = []
            for current_feature in xrange(self._n_features):
                # not exactly sure whether density=True is the right choice
                current_histogram = np.histogram(self._X[self._Y == current_class, current_feature],
                                                 n_bins_all_features[current_feature], density=True)
                histograms_per_class += [current_histogram]
            self._histograms += [histograms_per_class]

    def predict(self, x):
        """
        We use log likelihood instead of argmax formulation (for numerical stability)
        :param x:
        :return:
        """

        if self._histograms is None:
            raise Exception("train the classifier before you want to predict")

        y_predicted = np.ones(x.shape[0])*999

        # baeh dreifacher for loop in python...
        for j in xrange(x.shape[0]):
            best_result = -1
            best_class = -1

            # find argmax
            for i, current_class in enumerate(self.__classes):
                class_histograms = self._histograms[i]
                proba = np.log(self._priors[i])

                for feat_id in xrange(self._n_features):
                    my_value = x[i, feat_id]
                    bin_id = np.where(class_histograms[feat_id][1]<my_value)[0][-1]
                    proba += np.log(class_histograms[feat_id][0][bin_id])

                if proba > best_result:
                    best_result = proba
                    best_class = current_class

            y_predicted[j] = best_class
        return y_predicted

def dr(X, Y, n=2, method="ICAP"):
    featsel = feature_selection.filter_feature_selection.FilterFeatureSelection(X, Y, method=method)
    res = featsel.run(n)
    return X[:, res]


def load_data():
    f = h5py.File("data/small/test.h5")
    images_test = f["images"].value
    labels_test = f["labels"].value
    f.close()

    f = h5py.File("data/small/train.h5")
    images_train = f["images"].value
    labels_train = f["labels"].value
    f.close()

    return np.array(images_train), np.array(labels_train), np.array(images_test), np.array(labels_test)


def plot_3d_scatterplot(X_train, Y_train):
    values = np.unique(Y_train)
    assert len(values) == 2

    import matplotlib.pyplot as plt
    plt.title('Scatter Plot')
    plt.xlabel('X1')
    plt.ylabel('X2')

    size = 12

    plt.scatter(X_train[:, 0][Y_train == values[0]], X_train[:, 1][Y_train == values[0]], marker='x', c='r', s=size)
    plt.scatter(X_train[:, 0][Y_train == values[1]], X_train[:, 1][Y_train == values[1]], marker='o', c='b', s=size)
    plt.show()

if __name__ == "__main__":
    # load data
    train_X, train_Y, test_X, test_Y = load_data()

    # reshape do that we have 2 dimensional feature matrix
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))

    # select only 3's and 8's
    train_X = train_X[(train_Y == 3) | (train_Y == 8), :]
    train_Y = train_Y[(train_Y == 3) | (train_Y == 8)]
    test_X = test_X[(train_Y == 3) | (train_Y == 8), :]
    test_Y = test_Y[(train_Y == 3) | (train_Y == 8)]

    # feature selection
    selection_method = "mRMR"
    dr_train_X = dr(train_X, train_Y.astype("int"), 2, method=selection_method)
    dr_test_X = dr(test_X, test_Y.astype("int"), 2, method=selection_method)

    # plot the distribution in feature space to check if feature selection worked
    plot_3d_scatterplot(dr_train_X, train_Y)

