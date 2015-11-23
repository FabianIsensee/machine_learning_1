__author__ = "Fabian"
import sys
sys.path.append("../ex_2")
import numpy as np
import h5py
import feature_selection
from sklearn import cross_validation
import matplotlib.pyplot as plt


class NaiveBayes(object):
    def __init__(self):
        self._X = None
        self._Y = None
        self._histograms = None
        self._priors = None
        self._n_features = None
        self._classes = None

    def __find_nbins_auto(self):
        '''
        This function determines the best number of bins for each feature by using cross-validation. The error is
        calculated as the absolute difference between the density estimates of the train and test histograms

        :return: list (length n_features) of optimal number of bins
        '''
        candidate_bins = [2, 5, 10, 20, 50, 100, 200, 500, 1000] # should be sufficient for this exercise
        k_fold = 3
        cross_val = cross_validation.KFold(self._X.shape[0], n_folds=k_fold, shuffle=True)
        n_bins_optimal = []

        # oh yeah 4 nested for loops :^)
        for j in xrange(self._X.shape[1]):
            best_error = float("inf")
            best_n_bins = -1
            for nbins in candidate_bins:
                error = 0
                for kf in cross_val:
                    hist_train = np.histogram(self._X[kf[0], j], nbins, density=False)
                    hist_test = np.histogram(self._X[kf[1], j], nbins, density=False)
                    # actually we should use the same bin width instead of the same bin number, but in this case this
                    # results in approx. the same histograms
                    bin_width = hist_train[1][1] - hist_train[1][0]
                    for m in xrange(nbins):
                        error += (hist_train[0][m] / float(len(kf[0])) - hist_test[0][m] / float(len(kf[1])))**2 / bin_width
                error /= float(k_fold)
                print nbins, error
                if error < best_error:
                    best_n_bins = nbins
                    best_error = error
            n_bins_optimal += [best_n_bins]

        return n_bins_optimal

    def find_nbins_scott(self):
        n_bins_optimal = []
        for j in range(self._X.shape[1]):
            # minimum 2 bins...
            n_bins_optimal += [np.max((2, 3.5 * np.std(self._X[:, j]) / self._X.shape[0]**(1/3)))]
        return n_bins_optimal

    def train(self, x, y, n_bins = None):
        self._X = x
        self._Y = y

        self._classes = np.unique(self._Y)

        n_instances, self._n_features = self._X.shape

        self._histograms = []
        self._priors = []

        # find best number of bins
        if n_bins is not None:
            n_bins_all_features = np.ones(self._n_features) * n_bins
        else:
            n_bins_all_features = np.array(self.find_nbins_scott())

        # create histograms for each class and each feature
        for current_class in self._classes:
            self._priors += [np.sum(self._Y == current_class) / float(n_instances)]

            histograms_per_class = []
            for current_feature in xrange(self._n_features):
                # not exactly sure whether density=True is the right choice
                current_histogram = np.histogram(self._X[self._Y == current_class, current_feature],
                                                 n_bins_all_features[current_feature], density=True)
                histograms_per_class += [current_histogram]
            self._histograms += [histograms_per_class]

    def predict(self, x):
        """
        :param x:
        :return:
        """

        if self._histograms is None:
            raise Exception("train the classifier before you want to predict")

        y_predicted = np.ones(x.shape[0])*999

        # baeh dreifacher for loop in python...
        for j in xrange(x.shape[0]):
            best_result = -float("inf")
            best_class = -1

            # find argmax
            for i, current_class in enumerate(self._classes):
                class_histograms = self._histograms[i]
                proba = np.log(self._priors[i])

                for feat_id in xrange(self._n_features):
                    my_value = x[j, feat_id]

                    # min is necessary because sometimes a value can occur that is larger than the largest value in the training dataset
                    bin_id = np.min((np.where(class_histograms[feat_id][1] <= my_value)[0][-1], len(class_histograms[feat_id][0]) - 1))
                    proba += np.log(class_histograms[feat_id][0][bin_id])

                if proba > best_result:
                    best_result = proba
                    best_class = current_class

            y_predicted[j] = best_class
        return y_predicted

class GenerativeModel(NaiveBayes):
    def generate_instance_of_class(self, class_id, n):
        new_data = np.ones((n, self._n_features))
        idx = np.where(self._classes == class_id)[0]

        for j in xrange(self._n_features):
            bin_midpoints = self._histograms[idx][j][1][:-1] + np.diff(self._histograms[idx][j][1])/2.
            cdf = np.cumsum(self._histograms[idx][j][0])
            cdf /= cdf[-1]
            value = np.random.rand(n)
            value_bins = np.searchsorted(cdf, value)
            random_from_cdf = bin_midpoints[value_bins]
            new_data[:, j] = random_from_cdf
        return new_data

class Node():
    def __init__(self, data, n_instances_total, depth):
        self.data = data
        self._splitfeatureid = None
        self._threshold = None
        self._child1 = None
        self._child2 = None
        self.density = None
        self.isLeaf = False
        self._depth = depth
        self._n_instances_total = n_instances_total

        self.domain = self.get_domain_from_data(self.data)
        self.splitpos_functor = self._find_splitpos_random

    @staticmethod
    def get_domain_from_data(data):
        domain = np.array([np.min(data, axis = 0), np.max(data, axis = 0)]).transpose()
        return domain

    @staticmethod
    def calculate_volume(domain):
        edge_lengths = np.diff(domain, axis=1)
        return np.prod(edge_lengths)

    def _find_splitpos_middle(self, left_value, right_value):
        return (left_value + right_value) / 2.

    def _find_splitpos_random(self, left_value, right_value):
        randn = np.random.uniform()
        return left_value + randn * (right_value - left_value)

    def split(self, maxdepth = 10, mininstances = 10):
        if self._depth > maxdepth:
            self.isLeaf = True
            self.density = self._compute_density()
            return None

        best_split_feature = -1
        best_split_threshold = -1
        best_split_score = -1
        best_split_ids_left = None
        best_split_ids_right = None

        for j in xrange(self.data.shape[1]):
            sorted_values = np.argsort(self.data[:, j])
            for i in xrange(1, len(sorted_values)):
                # maximize (n_l / N)**2 * 1/v_l + (n_r / N)**2 * 1/v_r
                v_left = self.calculate_volume(self.get_domain_from_data(self.data[sorted_values[:i], :]))
                v_right = self.calculate_volume(self.get_domain_from_data(self.data[sorted_values[i:], :]))
                p_left = float(i) / float(self._n_instances_total)
                p_right = float(self.data.shape[0] - i) / float(self._n_instances_total)
                score = p_left**2 / v_left + p_right / v_right

                if score > best_split_score:
                    best_split_score = score
                    best_split_threshold = self.splitpos_functor(self.data[sorted_values[i-1]],
                                                                 self.data[sorted_values[i]])
                    best_split_feature = j
                    best_split_ids_left = sorted_values[:i]
                    best_split_ids_right = sorted_values[i:]

        if (len(best_split_ids_left) < mininstances) | (len(best_split_ids_right) < mininstances):
            self.isLeaf = True
            self.density = self._compute_density()
            return None

        self._child1 = Node(self.data[best_split_ids_left, :], self._n_instances_total, self._depth + 1)
        self._child2 = Node(self.data[best_split_ids_right, :], self._n_instances_total, self._depth + 1)
        self._threshold = best_split_threshold
        self._splitfeatureid = best_split_feature

        return self._child1, self._child2

    def _compute_density(self):
        # density = n / N / V
        volume = self.calculate_volume(self.domain)
        return float(self.data.shape[0]) / float(self._n_instances_total) / volume

    def find_likelihood(self, instance):
        if self.isLeaf:
            return self.density
        else:
            if instance[self._splitfeatureid] < self._threshold:
                return self._child1.find_likelihood(instance)
            else:
                return self._child2.find_likelihood(instance)




class DensityTree(object):
    def __init__(self):
        self._data = None
        self._priors = None
        self._n_features = None
        self._root_node = None

    def train(self, data, max_depth = 10, min_instances_per_node = 10):
        self._data = data
        node_stack = list()
        self._root_node = Node(self._data, self._data.shape[0], 0)
        node_stack.append(self._root_node)
        while len(node_stack) > 0:
            this_node = node_stack.pop()
            split_res = this_node.split(max_depth, min_instances_per_node)
            if split_res is not None:
                for node in split_res:
                    node_stack.append(node)

    def find_likelihood(self, data):
        assert(isinstance(self._root_node, Node), "Train the density tree first")
        for instance in data:
            self._root_node.find_likelihood(instance)

def dr(X, Y, n=2, method="ICAP"):
    featsel = feature_selection.filter_feature_selection.FilterFeatureSelection(X, Y, method=method)
    res = featsel.run(n)
    return X[:, res], res


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


def plot_2d_scatterplot(X_train, Y_train):
    values = np.unique(Y_train)
    assert len(values) == 2

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
    test_X = test_X[(test_Y == 3) | (test_Y == 8), :]
    test_Y = test_Y[(test_Y == 3) | (test_Y == 8)]

    # feature selection
    selection_method = "mRMR"
    dr_train_X, selected_ids = dr(train_X, train_Y.astype("int"), 2, method=selection_method)
    dr_test_X = test_X[:, selected_ids]

    # plot the distribution in feature space to check if feature selection worked
    # plot_2d_scatterplot(dr_test_X, test_Y)

    # train naive bayes
    nb = NaiveBayes()
    nb.train(dr_train_X, train_Y)

    # predict
    pred_Y = nb.predict(dr_test_X)
    accur = np.sum(pred_Y == test_Y) / float(len(test_Y))
    print "Accuracy with Naive Bayes (2 features): %f" % accur

    # generate 3's
    gm = GenerativeModel()
    gm.train(train_X, train_Y)
    n_new = 10
    new_threes = gm.generate_instance_of_class(3, n_new)
    for i in range(n_new):
        reshaped = np.reshape(new_threes[i, :], (9, 9))
        plt.imsave("generated_images/3_%02d.png" % i, reshaped, cmap="gray")
    # using a lot of fantasy, and maybe a bit of booye you can actually make out the three's

