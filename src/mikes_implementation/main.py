import numpy as np
from pprint import pprint
from math import log
import sys


class DecisionTree(object):
    def __init__(self, max_depth=10,
                       min_leaf=2,
                       continuous=False,
                       header=False,
                       bins=2):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.bins = bins
        self.header = header
        self.str_flt_index = []
        if continuous not in (False, 'static', 'dynamic'):
            print 'Invalid value for continuous:', continuous
            print 'Setting Continuous to "False"'
            continuous = False
        self.continuous = continuous

    def _str2flt(self, data, i=False, vals=False):
        if not vals:
            vals = list(np.unique(data))
            self.str_flt_index.append((i, vals))
        for i, val in enumerate(data):
            index = vals.index(val)
            data[i] = index
        return data

    def _checkFloatFeatures(self, data):
        data = data.T
        for i in range(len(data)):
            try:
                float(data[i][0])
            except:
                data[i] = self._str2flt(data[i], i=i)
        data = np.array(data, dtype=float)
        return data.T

    def _discretize(self, data):
        data -= np.min(data)
        tmp = np.array(data)
        max_val = np.max(data)/self.bins
        for i in range(self.bins):
            data[tmp >= max_val*i] = i
        return data

    def _discretizeData(self, data):
        data = data.T
        for i in range(len(data)):
            data[i] = self._discretize(data[i])
        return data.T

    def _normalizeData(self, data):
        data = data.T
        for i in range(len(data)):
            data[i] -= np.min(data[i])
            data[i] /= np.max(data[i])
        return data.T

    def _processData(self, data):
        data = np.array(data, dtype=object)
        if self.header:
            features = [(i, f) for i, f in enumerate(data[0,:-1])]
            klass = data[1:,-1]
            data = data[1:,:-1]
        else:
            features = [(i) for i in range(len(data[0]))]
            klass = data[:,-1]
            data = data[:,:-1]
        if self.continuous != False:
            data = self._checkFloatFeatures(data)
        if self.continuous == 'static':
            data = self._discretizeData(data)
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

    def _getFeatureDiscrete(self, data, features, klass):
        max_IG, max_feature = (0, None)
        for feature in features:
            IG = self._infoGain(data[:,feature[0]], klass)
            if IG > max_IG:
                max_IG, max_feature = (IG, feature)
        return max_feature

    def _getFeatureContinuous(self, data, features, klass):
        max_IG, max_feature = (0, None)
        for feature in features:
            tmp = np.array(data[:,feature[0]])
            tmp = self._discretize(tmp)
            IG = self._infoGain(tmp, klass)
            if IG > max_IG:
                max_IG, max_feature = (IG, feature)
        feature_data = data[:, max_feature[0]]
        size = (np.max(feature_data)-np.min(feature_data))/self.bins
        splits = [(i+1)*size+np.min(feature_data) for i in range(self.bins-1)]
        return max_feature, splits

    def _getFeature(self, data, features, klass):
        if self.continuous in (False, 'static'):
            return self._getFeatureDiscrete(data, features, klass), False
        else:
            return self._getFeatureContinuous(data, features, klass)

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
        split_feature, splits = self._getFeature(data, features, klass)
        if split_feature == None:
            return self._leafNode(klass)
        #del features[features.index(split_feature)] # remove split feature
        node['feature'] = split_feature
        node['children'] = {}
        i = split_feature[0]

        if self.continuous in (False, 'static'):
            vals = np.unique(data[:,i])
            for val in vals:
                index = data[:,i]==val
                args = (data[index], features, klass[index], depth+1)
                node['children'][val] = self._node(*args)
        else:
            for (mn, mx) in zip([-sys.maxint]+splits, splits+[sys.maxint]):
                index = data[:,i]
                index = (index>=mn) * (index<mx)
                args = (data[index], features, klass[index], depth+1)
                node['children'][(mn, mx)] = self._node(*args)

        return node

    def fit(self, data):
        data, features, klass = self._processData(data)
        self.tree = self._node(data, features, klass, 0)

    def _selectSolution(self, leaf_node):
        total = sum(p for p in leaf_node.values())
        rand = np.random.uniform(total)
        accum = 0
        for klass, prob in leaf_node.iteritems():
            if (accum + prob) >= rand:
                return klass
            accum += prob

    def _getClass(self, point, node):
        if not node['children']:
            return self._selectSolution(node['probs'])

        i = node['feature'][0]
        val = point[i]

        if self.continuous in (False, 'static'):
            next_node = node['children'][val]
        else:
            for child in node['children'].keys():
                if (val >= child[0]) and (val < child[1]):
                    next_node = node['children'][child]

        return self._getClass(point, next_node)

    def predict(self, test):
        test = np.array(test)
        if self.header:
            test = test[1:]
        klass = []

        if self.continuous != False:
            for item in self.str_flt_index:
                (i, vals) = item
                test[:,i] = self._str2flt(test[:,i], vals=vals)
            test = np.array(test, dtype=float)

        if self.continuous == 'static':
            test = self._discretizeData(test)

        for point in test:
            klass.append(self._getClass(point, self.tree))

        return np.array(klass)

    def getTree(self):
        if self.tree:
            return self.tree
        else:
            print 'No Tree Built'

import pydot
def getLabel(key):
    if isinstance(key, tuple):
        if key[0] == -sys.maxint:
            label = 'x<%.2f' % key[1]
        elif key[1] == sys.maxint:
            label = 'x>=%.2f' % key[0]
        else:
            label = '%.2f<=x<%.2f' % key
    else:
        return key
def exploreNode(node, graph, i):
    j = i
    for key, child in node['children'].iteritems():
        label = getLabel(key)
        j += 1
        if not child['children']:
            probs = child['probs']
            mx, name = (0, None)
            for k, v in probs.iteritems():
                if v > mx:
                    mx = v
                    name = k
            node = pydot.Node(str(j), label=str(name))
            graph.add_node(node)
            edge = pydot.Edge(str(i), str(j), label=label)
            graph.add_edge(edge)
        else:
            node = pydot.Node(str(j), label=child['feature'][1])
            graph.add_node(node)
            edge = pydot.Edge(str(i), str(j), label=label)
            graph.add_edge(edge)
            j = exploreNode(child, graph, j)
    return j

def printTree(tree):
    graph = pydot.Dot(graph_type='graph')
    node = pydot.Node('0', label=tree['feature'][1])
    graph.add_node(node)
    j = exploreNode(tree, graph, 0)
    graph.write_png('DT.png')


if __name__ == '__main__':
    with open('mpg_cars.csv') as f:
        data = f.readlines()
    data = map(lambda x: x.rstrip().split(','), data)
    data = np.array(data)
    data = np.hstack((data[:,1:], data[:,0:1]))
    test = np.array(data[:,:-1])
    true = np.array(data[1:,-1])

    model = DecisionTree(header=True, continuous='static', min_leaf=2, bins=3)
    model.fit(data)
    predict = model.predict(test)
    print 'Accuracy:',
    print float(np.sum(true == predict))/len(predict)
    tree = model.getTree()
    printTree(tree)
    #pprint(tree)

