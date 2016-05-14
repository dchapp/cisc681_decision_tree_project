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

def possible_values(data, feature):
    values = []
    for d in data:
        if d[feature] not in values:
            values.append(d[feature])
    return values

def bin_median(testing_data):
    features = list(testing_data[0].keys())
    features.remove('class')
    for f in features:
        all_values = possible_values(testing_data, f)
        if len(all_values) > 4:
            all_values = [float(x) for x in possible_values(testing_data, f)]
            median = np.median(all_values)
            for d in testing_data:
                if d[f] < median:
                    d[f] = "low"
                else:
                    d[f] = "high"

def bin_quartile(testing_data):
    features = list(testing_data[0].keys())
    features.remove("class")
    for f in features:
        all_values = possible_values(testing_data, f)
        if len(all_values) > 4:
            all_values = [float(x) for x in possible_values(testing_data, f)]
            q1 = np.percentile(all_values, 25)
            q2 = np.percentile(all_values, 50)
            q3 = np.percentile(all_values, 75)
            for d in testing_data:
                if d[f] < q1:
                    d[f] = "q1"
                elif d[f] < q2:
                    d[f] = "q2"
                elif d[f] < q3:
                    d[f] = "q3"
                else:
                    d[f] = "q4"


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
    print testing_data[0]
    bin_quartile(testing_data)
    print testing_data[0]

    

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
