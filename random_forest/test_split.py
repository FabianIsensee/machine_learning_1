__author__ = 'fabian'

import split_gini_cython
import numpy as np
import DecisionTree_optimized

feat = np.array([1,1,1,2,2,2,3,3,3,3], dtype=np.float32)
labels = np.array([0,0,0,1,1,1,1,0,0,1], dtype=np.int32)
classes = np.array([0,1], dtype=np.int32)
class_distrib = np.array([5,5], dtype=np.int32)

print split_gini_cython.split_gini(feat, labels, classes, class_distrib)

print DecisionTree_optimized.split_gini_new(feat, labels, class_distrib)


from sklearn import datasets, cross_validation

iris = datasets.load_iris()

x_tr, x_te, y_tr, y_te = cross_validation.train_test_split(iris.data, iris.target)
dt = DecisionTree_optimized.DecisionTree()
dt.train(x_tr, y_tr)
pred_y = dt.predict(x_te)
accur = np.sum(pred_y == y_te) / float(len(y_te)) * 100.
print(accur)