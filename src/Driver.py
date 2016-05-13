import sys

from DecisionTree import *
from Ingest import *

def main():
    #dt = DecisionTree()
    #dtn = DecisionTreeNode()
    training_data, features, classes = ingest_training_data(sys.argv[1])
    #print training_data
    dt = DecisionTree(training_data, features, classes)
    print dt.get_possible_values('mpg')
    print dt.get_fixed_value_subset('mpg', 'OK')
    print dt.get_class_counts(dt.training_data)
    print dt.entropy_of_subset(dt.training_data)
    for f in dt.features:
        print f + " " + str(dt.information_gain(f))
    #dt.bin_values_median('displacement')
    #dt.bin_values_quartile('displacement')
    print dt.training_data[0]
    print dt.features
    for f in dt.features:
        print f + ": " + str(len(dt.get_possible_values(f)))
    print dt.get_possible_values('displacement')
    dt.build_tree_id3()
    print dt.children
        
    

main()
