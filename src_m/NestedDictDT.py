import numpy as np
import pprint
from math import log


class DecisionTree(object):
    def __init__(self, max_depth=10, min_leaf=2):
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    def _processData(self, data):
        data = np.array(data, dtype=object)
        features = [(i, f) for i, f in enumerate(data[0,:-1])]
        klass = data[1:,-1]
        data = data[1:,:-1]
        return data, features, klass

    def _probs(self, klass, val):
        probs = float(np.sum(klass==val))
        probs = probs / len(klass)
        return probs

    def _leafCondition(self, data, features, klass, depth):
        if depth >= self.max_depth:
            return True
        if len(data) <= self.min_leaf:
            return True
        if len(np.unique(klass)) == 1:
            return True
        if len(features) == 0:
            return True
        return False

    def _entropy(self, klass):
        H = 0
        vals = np.unique(klass)
        if len(vals) == 1:
            return H
        for val in vals:
            prob = self._probs(klass, val)
            H -= prob * log(prob, 2)
        return H

    def _infoGain(self, data, klass):
        IG = self._entropy(klass)
        vals = np.unique(data)
        for val in vals:
            prob = self._probs(data,val)
            H = self._entropy(klass[data == val])
            IG -= prob*H
        return IG

    def _getFeature(self, data, features, klass):
        max_IG, max_feature = (0, None)
        for feature in features:
            IG = self._infoGain(data[:,feature[0]], klass)
            if IG > max_IG:
                max_IG, max_feature = IG, feature
        return max_feature

    def _leafNode(self, klass):
        leaf = {}
        leaf['children'] = False
        leaf['probs'] = {}
        vals = np.unique(klass)
        for val in vals:
            leaf['probs'][val] = self._probs(klass, val)
        return leaf

    def _node(self, data, features, klass, depth):
        if self._leafCondition(data, features, klass, depth):
            return self._leafNode(klass)

        node = {}
        split_feature = self._getFeature(data, features, klass)
        del features[features.index(split_feature)] # remove split feature
        node['feature'] = split_feature
        node['children'] = {}

        i = split_feature[0]
        vals = np.unique(data[:,i])
        for val in vals:
            index = data[:,i]==val
            args = (data[index], features, klass[index], depth+1)
            node['children'][val] = self._node(*args)

        return node

    def fit(self, data):
        data, features, klass = self._processData(data)
        self.tree = self._node(data, features, klass, 0)
        return self.tree

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f:
        data = f.readlines()

    data = map(lambda x: x.rstrip().split(','), data)
    data = np.array(data)

    model = DecisionTree()
    tree = model.fit(data)

    pprint.pprint(tree)


