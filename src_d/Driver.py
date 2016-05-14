import sys

from DecisionTree import *
from Ingest import *

def display(dt):
    print dt.children
    if not dt.children:
        print dt.klass
    else:
        for k in dt.children.keys():
            display(dt.children[k])

def main():
    training_data, features, classes = ingest_training_data(sys.argv[1])

    #print "Prior training data:"
    #print training_data

    #print training_data
    dt = DecisionTree(training_data, features, classes)

    #print "All training data:"
    #print dt.training_data
    #print "\n"
    
   # print dt.get_possible_values('mpg')
   # print dt.get_fixed_value_subset('mpg', 'OK')
   # print dt.get_class_counts(dt.training_data)
   # print dt.entropy_of_subset(dt.training_data)
   # for f in dt.features:
   #     print f + " " + str(dt.information_gain(f))
   # #dt.bin_values_median('displacement')
   # #dt.bin_values_quartile('displacement')
   # print dt.training_data[0]
   # print dt.features
   # for f in dt.features:
   #     print f + ": " + str(len(dt.get_possible_values(f)))
   # print dt.get_possible_values('displacement')
    
    
    dt.build_tree_id3()
    #display(dt) 

    testing_data, testing_features, testing_classes = ingest_training_data(sys.argv[2])
    num_test_instances = len(testing_data)
    num_correct = 0
    for t in testing_data:
        print "Actual class: " + str(t['class'])
        predicted_class = dt.classify(t)
        print "Predicted class: " + str(predicted_class)
        if predicted_class == t['class']:
            num_correct += 1

    accuracy = float(num_correct) / num_test_instances
    print "Accuracy = " + str(accuracy)


    

main()
