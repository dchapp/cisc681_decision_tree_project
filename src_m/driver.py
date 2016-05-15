from main import *
import sys
import numpy as np

with open(sys.argv[1]) as train_f:
    train_data = train_f.readlines()
train_data = map(lambda x: x.rstrip().split(','), train_data)
train_data = np.array(train_data)
train_data = np.hstack((train_data[:,1:], train_data[:,0:1]))

with open(sys.argv[2]) as test_f:
    test_data = test_f.readlines()
test_data = map(lambda x: x.rstrip().split(','), test_data)
test_data = np.array(test_data)
test_data = np.hstack((test_data[:,1:], test_data[:,0:1]))

if sys.argv[4] == "testing":
    true = np.array(test_data[1:,-1])
    test_data = np.array(test_data[:,:-1])
elif sys.argv[4] == "training":
    true = np.array(train_data[1:,-1])
    test_data = np.array(train_data[:,:-1])

model = DecisionTree(header=True, continuous=sys.argv[3], min_leaf=2, bins=3)
model.fit(train_data)
predict = model.predict(test_data)

print "Accuracy:",
print float(np.sum(true == predict)) / len(predict)


