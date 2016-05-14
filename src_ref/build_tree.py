
# Code from Chapter 12 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Code to run the decision tree on the Party dataset
import dtree

import sys

## Read training set and generate the decision tree
tree = dtree.dtree()
training_attributes,classes,features = tree.read_data(sys.argv[1])
t=tree.make_tree(training_attributes,
                 classes,
                 features)

## Print the tree on screen
tree.printTree(t,' ')

