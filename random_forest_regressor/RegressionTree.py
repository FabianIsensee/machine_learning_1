__author__ = 'fabian'
import numpy as np


class BaseNode(object):
    def __init__(self):
        self.isLeaf = False
        self._child_left = None
        self._child_right = None
        self._threshold = None
        self._splitfeature = None

    def split(self):
        raise NotImplementedError, "must be overridden by derived classes"

    def descent(self, feature_vec):



class RegressionNode(BaseNode):
    def __init__(self):
        super(RegressionNode).__init__()

    def split(self):
        raise NotImplementedError


class RegressionTree(object):
    def __init__(self):
        self._root_node = None

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

