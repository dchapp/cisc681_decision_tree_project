import numpy as np
import operator
from DecisionTreeNode import *

class DecisionTree(object):
    def __init__(self, training_data, features, classes):
        self.training_data = training_data
        self.features      = features
        self.classes       = classes
        self.children      = {}
        self.splitting_feature = None
        self.klass         = None
        #self.nodes         = [DecisionTreeNode(range(len(training_data)), None, i)]
    
    
    def build_tree_id3(self):
        for f in self.features:
            possible_values = self.get_possible_values(f)
            if len(possible_values) > 10:
                self.bin_values_median(f)
        self.build_tree_id3_helper()

    """
    Build the decision tree using the ID3 algorithm
    """
    def build_tree_id3_helper(self):
        class_counts = self.get_class_counts(self.training_data)
        #print "Class counts:"
        #print class_counts

        majority_class = max(class_counts.iteritems(), key=operator.itemgetter(1))[0]

        #print "DEBUG"
        #print self.features

        ### Case 1: All instances in subset have same class
        if class_counts[majority_class] == len(self.training_data):
            self.klass = majority_class
        ### Case 2: No features left to test
        elif len(self.features) == 0:
            self.klass = majority_class
        ### Case 3: Otherwise 
        else:
            ### Get feature that maximizes information gain
            gains = { f:0 for f in self.features }
            for f in self.features:
                gains[f] = self.information_gain(f)
            max_gain_feature = max(gains.iteritems(), key=operator.itemgetter(1))[0]

            #print "Max gain feature is: " + str(max_gain_feature)
            
            ### Assign that feature to the current node in the DT
            self.splitting_feature = max_gain_feature

            ### Make a child for each value the feature can take
            values = self.get_possible_values(max_gain_feature)
            for v in values:
                ### Get subset of training data for which this feature has value v
                subset = self.get_fixed_value_subset(max_gain_feature, v)
                
                #print "Subset for: " + str(max_gain_feature) + " = " + str(v)
                #print subset
                #print "\n"

                ### Exclude feature from future consideration
                new_features = list(self.features)
                new_features.remove(max_gain_feature)

                #print "DEBUG"
                #print new_features

                dt = DecisionTree(subset, new_features, self.classes)
                #dt.build_tree_id3_helper()
                self.children[v] = dt
                self.children[v].build_tree_id3_helper()


    """
    Classify a single instance of test data
    """
    def classify(self, test_instance):
        ### Descend to a leaf
        dt = self
        print dt.children
        #while type(dt.children.values()[0]) != str:
        while not dt.klass:
            splitting_value = test_instance[dt.splitting_feature]
            #print self.splitting_feature
            #print splitting_value
            dt = dt.children[splitting_value]
            #print dt.children
            #exit()
        ### Return class at leaf
        #return dt.children[test_instance[dt.splitting_feature]]
        return dt.klass


    """
    Compute the entropy of a single probability p.
    """
    def entropy(self, p):
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0

    """
    Compute the entropy of a subset of the training data
    """
    def entropy_of_subset(self, subset):
        e = 0
        num_instances = float(len(subset))
        class_counts  = self.get_class_counts(subset)
        for n in class_counts.values():
            e += self.entropy(n / num_instances)
        return e

    """
    Given a feature, get the set of values it takes 
    on in the training data.
    """
    def get_possible_values(self, feature):
        values = []
        for d in self.training_data:
            if d[feature] not in values:
                values.append(d[feature])
        return values

    """
    Bin continuous values based on median
    """
    def bin_values_median(self, feature):
        all_values = [float(x) for x in self.get_possible_values(feature)]
        median = np.median(all_values)
        for d in self.training_data:
            if d[feature] < median:
                d[feature] = "low"
            else:
                d[feature] = "high"

    """
    Bin continuous values into 4 bins based on quartiles

    """
    def bin_values_quartile(self, feature):
        all_values = [float(x) for x in self.get_possible_values(feature)]
        q1 = np.percentile(all_values, 25)
        q2 = np.percentile(all_values, 50)
        q3 = np.percentile(all_values, 75)
        stdv = np.std(all_values)
        for d in self.training_data:
            if d[feature] < q1:
                d[feature] = "q1"
            elif d[feature] < q2:
                d[feature] = "q2"
            elif d[feature] < q3:
                d[feature] = "q3"
            else:
                d[feature] = "q4"

    """
    Given a feature, get the subset of the training data
    for which that feature takes on a given value.
    """
    def get_fixed_value_subset(self, feature, value):
        subset = []
        for d in self.training_data:
            if d[feature] == value:
                subset.append(d)
        return subset

    """
    Given a subset of a subset of training data, count the 
    number of instances corresponding to each class.
    """
    def get_class_counts(self, subset):
        counts = { c:0 for c in self.classes }
        for d in subset:
            counts[d['class']] += 1
        return counts

    """
    Compute the information gain of a single feature 
    given a subset of the training data.
    """
    def information_gain(self, feature):
        ig = self.entropy_of_subset(self.training_data)
        possible_values = self.get_possible_values(feature)
        for v in possible_values:
            subset   = self.get_fixed_value_subset(feature, v)
            fraction = len(subset) / float(len(self.training_data))
            e        = self.entropy_of_subset(subset)
            ig -= fraction*e
        return ig



