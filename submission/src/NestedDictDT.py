import numpy as np
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

    # Convert string array to floats and track what float values correspond to
    # what strings
    def _str2flt(self, data, i=False, vals=False):
        if not vals:
            vals = list(np.unique(data))
            self.str_flt_index.append((i, vals))
        for i, val in enumerate(data):
            index = vals.index(val)
            data[i] = index
        return data

    # Check that all data in a 2D array is a float
    def _checkFloatFeatures(self, data):
        data = data.T
        for i in range(len(data)):
            try:
                float(data[i][0])
            except:
                data[i] = self._str2flt(data[i], i=i)
        data = np.array(data, dtype=float)
        return data.T

    # Discretize a 1D array of data
    def _discretize(self, data):
        data -= np.min(data)
        tmp = np.array(data)
        max_val = np.max(data)/self.bins
        for i in range(self.bins):
            data[tmp >= max_val*i] = i
        return data

    # Discretize a 2D array of data
    def _discretizeData(self, data):
        data = data.T
        for i in range(len(data)):
            data[i] = self._discretize(data[i])
        return data.T

    # Processes input data to be usable with DT
    def _processData(self, data):
        data = np.array(data, dtype=object)
        if self.header:
            features = [(i, f) for i, f in enumerate(data[0,:-1])]
            klass = data[1:,-1]
            data = data[1:,:-1]
        else:
            features = [(i, str(i)) for i in range(len(data[0,:-1]))]
            klass = data[:,-1]
            data = data[:,:-1]
        if self.continuous != False:
            # If continuous features, check if there are any categorical data
            data = self._checkFloatFeatures(data)
        if self.continuous == 'static':
            # Discretize data if method is static
            data = self._discretizeData(data)
        return data, features, klass

    # Gives the % of a class in an array
    def _probs(self, klass, val):
        probs = float(np.sum(klass==val))
        probs = probs / len(klass)
        return probs

    # Check if a leaf condition is met
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

    # Calculates entropy of an array
    def _entropy(self, klass):
        H = 0
        vals = np.unique(klass)
        if len(vals) == 1:
            return H
        for val in vals:
            prob = self._probs(klass, val)
            H -= prob * log(prob, 2)
        return H

    # Calculates the information gain
    def _infoGain(self, data, klass):
        IG = self._entropy(klass)
        vals = np.unique(data)
        for val in vals:
            prob = self._probs(data,val)
            H = self._entropy(klass[data == val])
            IG -= prob*H
        return IG

    # Method for getting best feature with discrete and static method
    def _getFeatureDiscrete(self, data, features, klass):
        max_IG, max_feature = (0, None)
        for feature in features:
            IG = self._infoGain(data[:,feature[0]], klass)
            if IG > max_IG:
                max_IG, max_feature = (IG, feature)
        return max_feature

    # Method for getting best continuous feature with dynamic method
    def _getFeatureContinuous(self, data, features, klass):
        max_IG, max_feature = (0, None)

        # Check each feature
        for feature in features:
            tmp = np.array(data[:,feature[0]])
            # Discretize data at each node - difference between static and dynamic
            tmp = self._discretize(tmp)
            IG = self._infoGain(tmp, klass)
            # Get the feature with the max Info Gain
            if IG > max_IG:
                max_IG, max_feature = (IG, feature)

        # Get feature data
        feature_data = data[:, max_feature[0]]

        # Decide where decision boundaries should occur
        size = (np.max(feature_data)-np.min(feature_data))/self.bins
        splits = [(i+1)*size+np.min(feature_data) for i in range(self.bins-1)]

        return max_feature, splits

    # Get the best feature to split on
    def _getFeature(self, data, features, klass):
        if self.continuous in (False, 'static'):
            return self._getFeatureDiscrete(data, features, klass), False
        else:
            return self._getFeatureContinuous(data, features, klass)

    # Returns a leaf node dictionary object
    def _leafNode(self, klass):
        leaf = {}
        leaf['children'] = False
        leaf['probs'] = {}

        # Get probability of each klass for leaf node
        vals = np.unique(klass)
        for val in vals:
            leaf['probs'][val] = self._probs(klass, val)

        return leaf

    # Node will add nodes to the tree until a stopping condition is met
    def _node(self, data, features, klass, depth):
        # Check if we should stop - return leaf node
        if self._leafCondition(data, features, klass, depth):
            return self._leafNode(klass)

        # Else create new node with children
        node = {}

        # Find best feature to split on
        split_feature, splits = self._getFeature(data, features, klass)

        # If no further progress can be made - return a leaf node
        if split_feature == None:
            return self._leafNode(klass)

        # Fill in node information
        node['feature'] = split_feature
        node['children'] = {}
        i = split_feature[0]

        # Creat the childrean nodes
        if self.continuous in (False, 'static'):
            # Discrete and static method for continuous features
            vals = np.unique(data[:,i])
            for val in vals:
                index = data[:,i]==val
                args = (data[index], features, klass[index], depth+1)
                node['children'][val] = self._node(*args)
        else:
            # Dynamic method for continuous features
            for (mn, mx) in zip([-sys.maxint]+splits, splits+[sys.maxint]):
                index = data[:,i]
                index = (index>=mn) * (index<mx)
                args = (data[index], features, klass[index], depth+1)
                node['children'][(mn, mx)] = self._node(*args)

        return node

    # Builds a decision tree with given data
    def fit(self, data):
        # Make sure info from previous tree is discarded
        self.str_flt_index = []

        # Process data
        data, features, klass = self._processData(data)

        # Build the tree
        self.tree = self._node(data, features, klass, 0)

    # Selects solution for leaf nodes
    def _selectSolution(self, leaf_node):
        # leaf_node contains probabilities of existing in each class
        total = sum(p for p in leaf_node.values())
        rand = np.random.uniform(total)
        accum = 0
        # Select the appropriate class based on given probabilities
        for klass, prob in leaf_node.iteritems():
            if (accum + prob) >= rand:
                return klass
            accum += prob

    # Traverses tree and gets class of a point
    def _getClass(self, point, node):
        # If we are at a leaf node, get the class
        if not node['children']:
            return self._selectSolution(node['probs'])

        # Get node information
        i = node['feature'][0]
        val = point[i]

        # Get next node to check
        if self.continuous in (False, 'static'):
            next_node = node['children'][val]
        else:
            # For dynamic method of continuous values, check range
            for child in node['children'].keys():
                if (val >= child[0]) and (val < child[1]):
                    next_node = node['children'][child]

        # Go to next node
        return self._getClass(point, next_node)

    # Predicts each test case using the built tree
    def predict(self, test):
        # place into np array, get rid of header if necessary
        test = np.array(test)
        if self.header:
            test = test[1:]

        klass = []

        # If continuous values, convert any categorical data to float values
        if self.continuous != False:
            for item in self.str_flt_index:
                (i, vals) = item
                test[:,i] = self._str2flt(test[:,i], vals=vals)
            test = np.array(test, dtype=float)

        # Discretize data if static method for continuous data
        if self.continuous == 'static':
            test = self._discretizeData(test)

        # Predict each point of the data
        for point in test:
            klass.append(self._getClass(point, self.tree))

        # Return an array of classes
        return np.array(klass)

    # Returns the tree object
    def getTree(self):
        if self.tree:
            return self.tree
        else:
            print 'No Tree Built'

