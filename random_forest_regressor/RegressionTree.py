__author__ = 'fabian'
import numpy as np
from random import sample
import IPython

class BaseNode(object):
    def __init__(self, features="sqrt", max_depth=None, min_instances=None):
        self.isLeaf = False
        self._child_left = None
        self._child_right = None
        self._threshold = None
        self._splitfeature = None
        self._prediction = None

        assert features in ["all", "sqrt"]
        self._features = features

        if max_depth is None:
            max_depth = 9999999999999
        if min_instances is None:
            min_instances = 1
        self._max_depth = max_depth
        self._min_instances = min_instances

class BaseTree(object):
    def __init__(self, features="sqrt", max_depth=None, min_instances=None):
        self._root_node = None
        self._features = features
        self._max_depth = max_depth
        self._min_instances = min_instances

    def descent(self, data_vec):
        curr_node = self._root_node
        while not curr_node.isLeaf:
            if data_vec[curr_node._splitfeature] < curr_node._threshold:
                curr_node = curr_node._child_left
            else:
                curr_node = curr_node._child_right
        return curr_node._prediction


class RegressionNode(BaseNode):
    def __init__(self, x, y, depth, features="sqrt", max_depth=None, min_instances=None):
        super(RegressionNode, self).__init__(features, max_depth, min_instances)
        self._x = x
        self._y = y
        self._depth = depth
        assert x.shape[0] == len(y)
        self._n_features = x.shape[1]
        self._n_instances = x.shape[0]

    def _check_constraints(self):
        if self._depth > self._max_depth:
            self._make_leaf()
            return False
        if self._n_instances <= self._min_instances:
            self._make_leaf()
            return False
        return True

    def _get_features_for_split(self):
        if self._features is "all":
            return np.arange(self._n_features)
        elif self._features is "sqrt":
            n_select = np.ceil(np.sqrt(self._n_features)).astype("int")
            return sample(np.arange(self._n_features), n_select)
        else:
            raise Exception("this should not happen...")

    def _make_leaf(self):
        self.isLeaf = True
        self._prediction = np.mean(self._y)

    def _find_best_split_of_feature(self, feat_id):
        #print "feat, depth: ", feat_id, self._depth
        #if feat_id == 12 and self._depth == 5:
        #    print "instances: ", self._n_instances
        best_residual = 99999999999.9
        best_split_pos = None
        idx_ordered = np.argsort(self._x[:, feat_id])
        curr_split_pos_left = 0
        curr_split_pos_right = 1

        while (curr_split_pos_left < self._n_instances - 1):
            while (curr_split_pos_left < self._n_instances - 1) and (self._x[idx_ordered[curr_split_pos_left], feat_id] ==
                   self._x[idx_ordered[curr_split_pos_right], feat_id]):
                curr_split_pos_left += 1
                curr_split_pos_right += 1

            if curr_split_pos_right > self._n_instances - 1:
                break

            '''if feat_id == 1 and self._depth == 5 and self._n_instances == 8:
                print curr_split_pos_left
                if curr_split_pos_left == 6:
                    IPython.embed()'''

            idx_left = idx_ordered[: curr_split_pos_left+1]
            idx_right = idx_ordered[curr_split_pos_left+1 :]
            residual = np.sum(np.square(self._y[idx_left] - np.mean(self._y[idx_left]))) + \
                       np.sum(np.square(self._y[idx_right] - np.mean(self._y[idx_right])))
            if residual < best_residual:
                best_residual = residual
                best_split_pos = curr_split_pos_left

            curr_split_pos_left += 1
            curr_split_pos_right += 1

        if best_split_pos is None:
            best_split_pos = -1
            split_threshold = None
        else:
            split_threshold = np.mean(self._x[idx_ordered[best_split_pos:best_split_pos+2], feat_id])
        return best_residual, split_threshold

    def split(self):
        best_split_value = 9999999999.9
        best_split_feature = None
        best_split_threshold = None

        for j in self._get_features_for_split():
            loc_best_split_score, loc_best_split_threshold = self._find_best_split_of_feature(j)

            if loc_best_split_score < best_split_value:
                best_split_value = loc_best_split_score
                best_split_feature = j
                best_split_threshold = loc_best_split_threshold

        if best_split_feature is None:
            self._make_leaf()
            return []

        all_ids = np.arange(self._n_instances)
        ids_left = all_ids[self._x[:, best_split_feature] <= best_split_threshold]
        ids_right = all_ids[self._x[:, best_split_feature] > best_split_threshold]
        '''if len(ids_left) == 0 or len(ids_right) == 0:
            IPython.embed()'''

        returned_children = []

        self._child_left = RegressionNode(self._x[ids_left, :], self._y[ids_left], self._depth + 1, self._features, self._max_depth, self._min_instances)
        self._child_right = RegressionNode(self._x[ids_right, :], self._y[ids_right], self._depth + 1, self._features, self._max_depth, self._min_instances)

        self._splitfeature = best_split_feature
        self._threshold = best_split_threshold

        if self._child_left._check_constraints():
            returned_children.append(self._child_left)
        if self._child_right._check_constraints():
            returned_children.append(self._child_right)

        return returned_children

class RegressionTree(BaseTree):
    def __init__(self, features="sqrt", max_depth=None, min_instances=None):
        super(RegressionTree, self).__init__(features, max_depth, min_instances)

    def fit(self, x, y):
        self._root_node = RegressionNode(x, y, 0, self._features, self._max_depth, self._min_instances)
        stack = [self._root_node]
        while len(stack) > 0:
            curr_node = stack.pop(0)
            new_nodes = curr_node.split()
            stack += new_nodes

    def predict(self, x):
        assert self._root_node is not None

        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        y_pred = np.zeros(x.shape[0])

        for i, data_vec in enumerate(x):
            y_pred[i] = self.descent(data_vec)

        return y_pred


if __name__ == "__main__":
    from sklearn import datasets
    boston = datasets.load_boston()
    data = boston.data
    target = boston.target

    from sklearn import cross_validation
    x_tr, x_te, y_tr, y_te = cross_validation.train_test_split(data, target, random_state=0)

    rt = RegressionTree(max_depth=5, features="all", min_instances=None)
    rt.fit(x_tr, y_tr)

    y_pred = rt.predict(x_te)

    residuals = np.sum(np.square(y_pred - y_te))
    print residuals

    from sklearn.tree import DecisionTreeRegressor
    dtr = DecisionTreeRegressor(max_depth=5)
    dtr.fit(x_tr, y_tr)
    y_pred = dtr.predict(x_te)
    residuals = np.sum(np.square(y_pred - y_te))
    print residuals
