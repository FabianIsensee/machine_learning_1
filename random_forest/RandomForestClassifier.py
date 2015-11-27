__author__ = "Fabian Isensee"

import numpy as np
from DecisionTree import DecisionTree



class RandomForestClassifier(object):
    def __init__(self, n_estimators=100, use_features="sqrt", max_depth=None, min_instances=None):
        self._n_estimators = n_estimators
        self._use_features = use_features
        self._max_depth = max_depth
        self._min_instances = min_instances
        self._trees = []
        self._classes = None

    def train(self, x, y):
        self._classes = np.unique(y)
        all_idx = np.arange(x.shape[0])
        for i in xrange(self._n_estimators):
            # print "training tree %d" % i
            dt = DecisionTree(self._use_features, self._max_depth, self._min_instances)
            idx = np.random.choice(all_idx, x.shape[0], replace=True)
            dt.train(x[idx, :], y[idx])
            self._trees += [dt]

    def predict(self, x):
        assert len(self._trees) > 0, "train the Decision Tree first..."

        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        pred_y = np.zeros(x.shape[0])

        for i in xrange(x.shape[0]):
            votes = np.zeros(len(self._classes))
            for tree in self._trees:
                res_tree = tree.predict(x[i, :])
                votes[self._classes == res_tree] += 1
            pred_y[i] = self._classes[np.argmax(votes)]

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

    print "\nnow do rf"
    rf = RandomForestClassifier(100, "sqrt", None, None)
    print "start training rf. timestamp: %f" % (os.times()[4]-start_time)
    rf.train(train_x, train_y)

    print "start predicting rf. timestamp: %f" % (os.times()[4]-start_time)
    pred_y_rf = rf.predict(test_x)
    print "predicting done. timestamp: %f" % (os.times()[4]-start_time)
    accur = np.sum(pred_y_rf == test_y) / float(len(test_y))
    print "accuracy rf: ", accur

    import sklearn.ensemble
    print "\nnow sklearn rf"
    rf_sklearn = sklearn.ensemble.RandomForestClassifier(n_estimators = 100)
    print "start training rf sklearn. timestamp: %f" % (os.times()[4]-start_time)
    rf_sklearn.fit(train_x, train_y)
    print "start predicting rf sklearn. timestamp: %f" % (os.times()[4]-start_time)
    pred_y_rfs = rf_sklearn.predict(test_x)
    print "predicting done. timestamp: %f" % (os.times()[4]-start_time)
    accur = np.sum(pred_y_rfs == test_y) / float(len(test_y))
    print "accuracy rf sklearn: ", accur