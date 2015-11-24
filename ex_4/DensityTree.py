import numpy as np
import IPython

class Node(object):
    def __init__(self, data, n_instances_total, depth, domain=None):
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
        domain = np.array([np.min(self._data, axis=0), np.max(self._data, axis=0)]).transpose()
        return domain

    @staticmethod
    def _calculate_volume(domain):
        edge_lengths = np.diff(domain, axis=1)
        return np.prod(edge_lengths)

    def _calculate_density(self):
        volume = self._calculate_volume(self._domain)
        density = float(self._n_instances) / float(self._n_instances_total) / float(volume)
        return density

    @staticmethod
    def _find_splitpos_middle(value_left, value_right):
        return float(value_left + value_right) / 2.

    def _make_leaf(self):
        self.isLeaf = True

    def _check_node_for_constraints(self, node, min_instances_per_node, max_depth):
        if (node._n_instances < min_instances_per_node) or \
           (node._depth > max_depth):
            node._make_leaf()
        return node

    def split(self, min_instances_per_node=10, max_depth=10):
        if self.isLeaf:
            return None

        best_split_score = -1
        best_split_threshold = -1
        best_split_feature = -1
        best_split_ids_left = None
        best_split_ids_right = None
        best_split_domain_left = None
        best_split_domain_right = None

        for j in xrange(self._n_features):
            left_pos = 0
            idx_sorted_by_feature = np.argsort(self._data[:, j])

            while left_pos < (self._n_instances - 1):
                while (left_pos < self._n_instances - 1) and \
                      (self._data[idx_sorted_by_feature[left_pos], j] == self._data[idx_sorted_by_feature[left_pos + 1], j]):
                    left_pos += 1

                right_pos = left_pos + 1

                if right_pos >= self._n_instances:
                    break

                value_left = self._data[idx_sorted_by_feature[left_pos], j]
                value_right = self._data[idx_sorted_by_feature[right_pos], j]
                splitpos = self._find_splitpos_middle(value_left, value_right)
                domain_left = np.array(self._domain)
                domain_right = np.array(self._domain)
                domain_left[j, 1] = splitpos
                domain_right[j, 0] = splitpos
                v_left = self._calculate_volume(domain_left)
                v_right = self._calculate_volume(domain_right)
                if (v_left == 0) or (v_right == 0):
                    IPython.embed()

                proportion_left = float(value_left + 1) / float(self._n_instances_total)
                proportion_right = float(self._n_instances - value_left - 1) / float(self._n_instances_total)

                score = np.square(proportion_left) / v_left + np.square(proportion_right) / v_right

                if score > best_split_score:
                    best_split_feature = j
                    best_split_score = score
                    best_split_threshold = splitpos
                    best_split_ids_left = idx_sorted_by_feature[:(left_pos + 1)]
                    best_split_ids_right = idx_sorted_by_feature[(left_pos + 1):]
                    best_split_domain_left = domain_left
                    best_split_domain_right = domain_right

                left_pos = right_pos


        if best_split_ids_right is None:
            self._make_leaf()
            return None
        else:
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
        if len(x.shape) == 0:
            x = x.reshape((1, -1))

        densities = np.zeros(x.shape[0])

        for i in xrange(x.shape[0]):
            densities[i] = self._root_node.find_density(x[i, :])

        return densities

class DensityEstimationTreeClassifier(object):
    def __init__(self):
        raise NotImplementedError


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn import cross_validation

    iris = datasets.load_iris()
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(iris.data, iris.target)

    det = DensityEstimationTree()
    det.train(train_X)

    print det.get_density(test_X)