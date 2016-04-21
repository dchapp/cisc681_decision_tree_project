import numpy as np

class DecisionTree(object):
    def __init__(self):
        return 0

    """
    Compute the entropy of a single probability p.
    """
    def entropy(p):
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0

    """
    Given a feature, get the set of values it takes 
    on in the training data.
    """
    def get_possible_values(data, feature):
        values = []
        for d in data:
            if d[feature] not in values:
                values.append(d[feature])
        return values

    """
    Given a feature, get the subset of the training data
    for which that feature takes on a given value.
    """
    def get_fixed_value_subset(data, feature, value):
        subset = []
        for d in data:
            if d[feature] == value:
                subset.append(d)
        return subset

    

    """
    Given a subset of the training data, count the 
    number of instances corresponding to each class.
    """
    def get_class_counts(data, classes):
        counts = { c:0 for c in classes }
        for d in data:
            counts[d['class']] += 1
        return counts

    """
    Compute the information gain of a feature given 
    a subset of the training data.
    """
    def information_gain(data, classes, feature):
        ig = 0
        num_instances = len(data)



