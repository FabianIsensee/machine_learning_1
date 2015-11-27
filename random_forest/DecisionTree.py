__author__ = "Fabian Isensee"
import numpy as np
from random import sample
import IPython


class DecisionTreeNode(object):
    def __init__(self, data, labels, use_features = "all", depth = 0):
        self.data = data
        self.labels = labels
        self._classes = np.unique(self.labels)
        assert self.data.shape[0] == len(self.labels), "data and labels must have the same number of instances:\n" \
                                                             "data: %d\n" \
                                                             "labels: %d" % (self.data.shape[0], len(self.labels))

        self._threshold = None
        self._splitFeatureID = None

        self.isLeaf = False

        self._n_instances, self._n_features = self.data.shape

        self._depth = depth

        assert use_features in ["all", "sqrt"], "use_features must be either \"all\" or \"sqrt\""
        if use_features == "all":
            self._get_features_for_split = self.__get_features_for_split_all
        else:
            self._get_features_for_split = self.__get_features_for_split_sqrt
        self._use_features = use_features

        self.__predicted_class = -1

        self._child_left = None
        self._child_right = None

    def __get_features_for_split_all(self):
        return np.arange(self._n_features)

    def __get_features_for_split_sqrt(self):
        n_select = np.ceil(np.sqrt(self._n_features)).astype("int")
        return sample(self.__get_features_for_split_all(), n_select)

    def _find_predicted_class(self):
        predicted_probas = np.zeros(len(self._classes)).astype("float")
        for k, curr_class in enumerate(self._classes):
            predicted_probas[k] = float(np.sum(self.labels == curr_class))
        predicted_probas /= float(self._n_instances)
        return self._classes[np.argmax(predicted_probas)]

    def _make_leaf(self):
        self.__predicted_class = self._find_predicted_class()
        self.isLeaf = True

    def find_prediction(self, data_vec):
        if self.isLeaf:
            return self.__predicted_class
        else:
            if data_vec[self._splitFeatureID] < self._threshold:
                return self._child_left.find_prediction(data_vec)
            else:
                return self._child_right.find_prediction(data_vec)

    def check_constraints_violated(self, max_depth=10, min_instances=10):
        if len(self._classes) == 1:
            self._make_leaf()
            return True
        if (min_instances is not None) and (self._n_instances < min_instances):
            self._make_leaf()
            return True
        if (max_depth is not None) and (self._depth > max_depth):
            self._make_leaf()
            return True
        return False

    def split_node(self, max_depth=10, min_instances=10):
        features_for_split = self._get_features_for_split()

        # gini coefficient must be minimized by the split
        best_split_score = float("inf")
        best_split_feature = None
        best_split_threshold = None
        best_split_indices_left = None
        best_split_indices_right = None

        for j in features_for_split:
            idx_sorted_by_feature = np.argsort(self.data[:, j])
            left_idx = 0

            while left_idx < (self._n_instances - 1):
                right_idx = left_idx + 1
                while (left_idx < (self._n_instances - 1)) and \
                        (self.data[idx_sorted_by_feature[left_idx], j] ==
                         self.data[idx_sorted_by_feature[right_idx], j]):
                    left_idx += 1
                    right_idx += 1

                if right_idx > self._n_instances - 1:
                    break

                n_idx_left = left_idx + 1
                n_idx_right = self._n_instances - n_idx_left
                p_left = float(n_idx_left) / float(self._n_instances)
                p_right = 1. - p_left

                gini_coeff = 0.

                for curr_class in self._classes:
                    p_k_left = float(np.sum(self.labels[idx_sorted_by_feature[:(left_idx + 1)]] == curr_class)) / \
                        float(n_idx_left)
                    p_k_right = float(np.sum(self.labels[idx_sorted_by_feature[(left_idx + 1):]] == curr_class)) / \
                        float(n_idx_right)
                    gini_coeff += p_left * p_k_left * (1. - p_k_left) + p_right * p_k_right * (1. - p_k_right)

                if best_split_score > gini_coeff:
                    best_split_score = gini_coeff
                    best_split_threshold = np.mean(self.data[idx_sorted_by_feature[[left_idx, right_idx]], j])
                    best_split_feature = j
                    best_split_indices_left = idx_sorted_by_feature[:(left_idx + 1)]
                    best_split_indices_right = idx_sorted_by_feature[(left_idx + 1):]

                left_idx += 1

        # this can rarely happen if the only feature where the feature values differ is not selected. then there is no
        # valid split that can be found
        if best_split_feature is None:
            self._make_leaf()
            return []

        self._threshold = best_split_threshold
        self._splitFeatureID = best_split_feature

        self._child_left = DecisionTreeNode(self.data[best_split_indices_left, :],
                                            self.labels[best_split_indices_left],
                                            use_features=self._use_features,
                                            depth=self._depth + 1)

        self._child_right = DecisionTreeNode(self.data[best_split_indices_right, :],
                                             self.labels[best_split_indices_right],
                                             use_features=self._use_features,
                                             depth=self._depth + 1)

        returned_children = []
        for child in [self._child_left, self._child_right]:
            if not child.check_constraints_violated(max_depth, min_instances):
                returned_children.append(child)

        return returned_children



class DecisionTree(object):
    def __init__(self, use_features="sqrt", max_depth=None, min_instances=None):
        self._use_features = use_features
        self._max_depth = max_depth
        self._min_instances = min_instances
        self._root_node = None

    def train(self, x, y):
        self._root_node = DecisionTreeNode(x, y, self._use_features, 0)
        node_stack = [self._root_node]

        while len(node_stack) > 0:
            this_node = node_stack.pop()
            res = this_node.split_node(self._max_depth, self._min_instances)
            node_stack += res

    def predict(self, x):
        assert self._root_node is not None, "train the Decision Tree first..."

        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        pred_y = np.zeros(x.shape[0])

        for i in xrange(x.shape[0]):
            pred_y[i] = self._root_node.find_prediction(x[i, :])

        return pred_y


if __name__ == "__main__":
    from sklearn import datasets, cross_validation
    import os
    iris = datasets.load_iris()

    start_time = os.times()[4]
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(iris.data, iris.target)

    dt = DecisionTree()
    print "start training decision tree. timestamp: %f" % (os.times()[4]-start_time)
    dt.train(train_x, train_y)

    print "start predicting decision tree. timestamp: %f" % (os.times()[4]-start_time)
    pred_y = dt.predict(test_x)
    print "predicting done. timestamp: %f" % (os.times()[4]-start_time)
    accur = np.sum(pred_y == test_y) / float(len(test_y))
    print "accuracy: ", accur
