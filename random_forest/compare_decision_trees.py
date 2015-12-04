from DecisionTree import DecisionTree
from DecisionTree_optimized import DecisionTree as DecisionTree_opt
import numpy as np
from sklearn import datasets, cross_validation
import os

iris = datasets.load_digits()

train_x, test_x, train_y, test_y = cross_validation.train_test_split(iris.data, iris.target)

start_time = os.times()[4]
dt = DecisionTree('all')
print "start training regular decision tree. timestamp: %f" % (os.times()[4]-start_time)
dt.train(train_x, train_y)

print "start predicting decision tree. timestamp: %f" % (os.times()[4]-start_time)
pred_y = dt.predict(test_x)
print "predicting done. timestamp: %f" % (os.times()[4]-start_time)
accur = np.sum(pred_y == test_y) / float(len(test_y))
print "accuracy: ", accur


start_time = os.times()[4]
dt = DecisionTree_opt('all')
print "start training new decision tree. timestamp: %f" % (os.times()[4]-start_time)
dt.train(train_x, train_y)

print "start predicting decision tree. timestamp: %f" % (os.times()[4]-start_time)
pred_y = dt.predict(test_x)
print "predicting done. timestamp: %f" % (os.times()[4]-start_time)
accur = np.sum(pred_y == test_y) / float(len(test_y))
print "accuracy: ", accur


