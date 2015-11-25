import numpy as np
import IPython

class Node(object):
    def __init__(self, data, n_instances_total, depth, domain=None):
        """
        the node class is used to construct density trees. Each node stores the data associated to it, the total numner
        of instances that are in the dataset (the whoile dataset, not the node specific data), its depth and its domain
        :param data:
        :param n_instances_total:
        :param depth:
        :param domain:
        :return:
        """
        self._split_feature_id = None
        self._split_threshold = None
        self._child1 = None
        self._child2 = None
        self._depth = depth
        self._n_instances_total = n_instances_total
        self._data = data
        self._n_features = self._data.shape[1]
        self._n_instances = self._data.shape[0]
        self.isLeaf = False

        if domain is None:
            self._domain = self._calculate_domain()
        else:
            self._domain = domain

    def _calculate_domain(self):
        """
        This is only used in the root node to calculate the initial domain of the data. Every intermediate/leaf node
        will then get its domain via __init__ (required to encode splitpositions correctly)
        :return:
        """
        domain = np.array([np.min(self._data, axis=0), np.max(self._data, axis=0)]).transpose()
        return domain

    @staticmethod
    def _calculate_volume(domain):
        """
        Calculates the volume spanned by a domain
        :param domain:
        :return:
        """
        edge_lengths = np.diff(domain.astype("float"), axis=1)
        return np.prod(edge_lengths)

    def _calculate_density(self):
        """
        Calculates the estimated density of a node using its instance count and domain
        :return:
        """
        volume = self._calculate_volume(self._domain)
        density = float(self._n_instances) / float(self._n_instances_total) / float(volume)
        return density

    @staticmethod
    def _find_splitpos_middle(value_left, value_right):
        """
        simply returs the mean of two values
        :param value_left:
        :param value_right:
        :return:
        """
        return np.mean([value_left, value_right])

    def _make_leaf(self):
        """
        marks whether a node is a leaf
        :return:
        """
        self.isLeaf = True

    def _check_node_for_constraints(self, node, min_instances_per_node, max_depth):
        """
        checks constraints and transforms itself into leaf if constraints are violated
        :param node:
        :param min_instances_per_node:
        :param max_depth:
        :return:
        """
        if (node._n_instances < min_instances_per_node) or \
           (node._depth > max_depth):
            node._make_leaf()
        return node

    def split(self, min_instances_per_node=10, max_depth=10):
        """
        the split function. More comments provided below

        :param min_instances_per_node:
        :param max_depth:
        :return:
        """
        if self.isLeaf:
            return None

        best_split_score = -1
        best_split_threshold = -1
        best_split_feature = -1
        best_split_ids_left = None
        best_split_ids_right = None
        best_split_domain_left = None
        best_split_domain_right = None

        # iterate over all features
        for j in xrange(self._n_features):
            left_pos = 0
            # argsort features by value
            idx_sorted_by_feature = np.argsort(self._data[:, j])

            # iterate over all possible splits
            while left_pos < (self._n_instances - 1):
                # if we have, say [1, 1, 1, 2, 3] as the sorted feature vector, we want the first split tp be between the
                # 1 and the 2, not between two 1's, therefore increment the left index until value(left)!=value(left+1)
                while (left_pos < self._n_instances - 1) and \
                      (self._data[idx_sorted_by_feature[left_pos], j] == self._data[idx_sorted_by_feature[left_pos + 1], j]):
                    left_pos += 1

                right_pos = left_pos + 1

                if right_pos >= self._n_instances:
                    break

                # so far, left_pos and right_pos are only indices in the argsorted array. Now we need the actual
                # feature values at these positions
                value_left = self._data[idx_sorted_by_feature[left_pos], j]
                value_right = self._data[idx_sorted_by_feature[right_pos], j]

                # compute split position as the middle between left and right value
                splitpos = self._find_splitpos_middle(value_left, value_right)

                # adapt the dimains of the potential child nodes
                domain_left = np.array(self._domain)
                domain_right = np.array(self._domain)
                domain_left[j, 1] = splitpos
                domain_right[j, 0] = splitpos
                # calculate the volume of the children using their domain
                volume_left = self._calculate_volume(domain_left)
                volume_right = self._calculate_volume(domain_right)
                if (volume_left == 0) or (volume_right == 0):
                    # this should never happen. If it does then there is a mistake somewhere
                    IPython.embed()

                # find how many instances would be left or right when this split is done
                instances_left = float(left_pos + 1)
                instances_right = float(self._n_instances - left_pos - 1)

                # compute the score of the split
                score = np.square(instances_left) / volume_left + np.square(instances_right) / volume_right

                # update score if it is better than the previous one
                if score > best_split_score:
                    best_split_feature = j
                    best_split_score = score
                    best_split_threshold = splitpos
                    best_split_ids_left = idx_sorted_by_feature[:(left_pos + 1)]
                    best_split_ids_right = idx_sorted_by_feature[(left_pos + 1):]
                    best_split_domain_left = domain_left
                    best_split_domain_right = domain_right

                # go to next possible split position
                left_pos = right_pos

        # if no valid split could be found (for example for [3, 3, 3, 3, 3] no valid split is possible), make leaf
        if best_split_ids_right is None:
            self._make_leaf()
            return None
        else:
            # if not, create child nodes
            print "\ncreating two child nodes..."
            print "splitpos was", best_split_threshold
            print "feature id", best_split_feature
            print "child1 size: ", len(best_split_ids_left), "\t child2 size: ", len(best_split_ids_right)
            print "child nodes depth:", self._depth+1

            self._split_threshold = best_split_threshold
            self._split_feature_id = best_split_feature
            child1 = Node(self._data[best_split_ids_left, :], self._n_instances_total, self._depth + 1,
                          best_split_domain_left)
            child2 = Node(self._data[best_split_ids_right, :], self._n_instances_total, self._depth + 1,
                          best_split_domain_right)
            child1 = self._check_node_for_constraints(child1, min_instances_per_node, max_depth)
            child2 = self._check_node_for_constraints(child2, min_instances_per_node, max_depth)
            self._child1 = child1
            self._child2 = child2
            return self._child1, self._child2

    def find_density(self, data_vec):
        """
        descends recursively along the nodes until a leaf is reached. then, return density of leaf
        :param data_vec:
        :return:
        """
        if self.isLeaf:
            return self._calculate_density()
        else:
            if data_vec[self._split_feature_id] < self._split_threshold:
                return self._child1.find_density(data_vec)
            else:
                return self._child2.find_density(data_vec)

class DensityEstimationTree(object):
    def __init__(self):
        self._root_node = None

    def train(self, x, min_instances_per_node=10, max_depth=10):
        """
        creates a stack. new nodes are placed on the stack, in each iteration, one node is taken from the stack and
        split
        :param x:
        :param min_instances_per_node:
        :param max_depth:
        :return:
        """
        self._root_node = Node(x, x.shape[0], 0)

        node_stack = []
        node_stack.append(self._root_node)

        while len(node_stack) > 0:
            current_node = node_stack.pop()
            split_result = current_node.split(min_instances_per_node=min_instances_per_node, max_depth=max_depth)
            if split_result is not None:
                for new_node in split_result:
                    node_stack.append(new_node)

    def get_density(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        densities = np.zeros(x.shape[0])
        for i in xrange(x.shape[0]):
            densities[i] = self._root_node.find_density(x[i, :])

        return densities

class DensityEstimationTreeClassifier(object):
    def __init__(self):
        self._classes = None
        self._class_trees = None
        self._priors = []

    def train(self, x, y, min_instances_per_node=10, max_depth=10):
        self._classes = np.unique(y)
        self._priors = np.zeros(len(self._classes))
        self._class_trees = []

        for k, current_class in enumerate(self._classes):
            indexes_of_class = np.where(y == current_class)[0]
            self._priors[k] = float(len(indexes_of_class)) / float(x.shape[1])
            this_tree = DensityEstimationTree()
            this_tree.train(x[indexes_of_class, :], min_instances_per_node, max_depth)
            self._class_trees += [this_tree]

    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        pred_y = np.zeros(x.shape[0])

        for i in xrange(x.shape[0]):
            scores = []
            for k in xrange(len(self._classes)):
                scores += [self._class_trees[k].get_density(x[i, :]) * self._priors[k]]
            pred_y[i] = self._classes[np.argmax(scores)]
        return pred_y


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn import cross_validation

    iris = datasets.load_iris()
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(iris.data, iris.target)

    det = DensityEstimationTree()
    det.train(train_X)

    print det.get_density(test_X)
    print "\n"

    dtc = DensityEstimationTreeClassifier()
    dtc.train(train_X, train_Y, 10, 10)
    pred_Y = dtc.predict(test_X)
    accur = float(np.sum(pred_Y == test_Y)) / float(len(test_Y))
    print "accuracy on test: %f" % accur