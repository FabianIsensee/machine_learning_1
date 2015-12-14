__author__ = 'fabian'
from RegressionTree import RegressionTree
import numpy as np

class BaseForest(object):
    def __init__(self, n_estimators=100, use_features="sqrt", max_depth=None, min_instances=None):
        self._n_estimators = n_estimators
        self._use_features = use_features
        self._max_depth = max_depth
        self._min_instances = min_instances
        self._trees = []


class RandomForestRegressor(BaseForest):
    def __init__(self, n_estimators=100, use_features="sqrt", max_depth=None, min_instances=None):
        super(RandomForestRegressor, self).__init__(n_estimators, use_features, max_depth, min_instances)

    def fit(self, x, y):
        all_idx = np.arange(x.shape[0])
        for i in xrange(self._n_estimators):
            # print "training tree %d" % i
            dt = RegressionTree(self._use_features, self._max_depth, self._min_instances)
            idx = np.random.choice(all_idx, x.shape[0], replace=True)
            dt.fit(x[idx, :], y[idx])
            self._trees += [dt]

    def predict(self, x):
        assert len(self._trees) > 0, "train the Decision Tree first..."
        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        pred_y = np.zeros(x.shape[0])

        for i, data_vec in enumerate(x):
            tree_predictions = np.zeros(self._n_estimators)
            for j, tree in enumerate(self._trees):
                tree_predictions[j] = tree.predict(x[i, :])
            pred_y[i] = np.mean(tree_predictions)

        return pred_y


if __name__ == "__main__":
    from sklearn import datasets
    boston = datasets.load_boston()
    data = boston.data
    target = boston.target

    from sklearn import cross_validation
    x_tr, x_te, y_tr, y_te = cross_validation.train_test_split(data, target, random_state=0)

    rfr = RandomForestRegressor(n_estimators=100, max_depth=5, use_features="sqrt", min_instances=None)
    rfr.fit(x_tr, y_tr)

    y_pred = rfr.predict(x_te)

    residuals = np.sum(np.square(y_pred - y_te))
    print residuals

    from sklearn.ensemble import RandomForestRegressor as RFR
    rfr_sk = RFR(max_depth=5, n_estimators=100, max_features="sqrt")
    rfr_sk.fit(x_tr, y_tr)
    y_pred = rfr_sk.predict(x_te)
    residuals = np.sum(np.square(y_pred - y_te))
    print residuals
