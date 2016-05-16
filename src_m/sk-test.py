from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sys

with open(sys.argv[1]) as train_f:
    data = train_f.readlines()
    data = map(lambda x: x.rstrip().split(','), data)
    data = np.array(data)
    data = np.hstack((data[:,1:], data[:,0:1]))

target = np.array(data[1:,-1])
data = np.array(data[1:,:-1])

clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf=3)
recall = cross_val_score(clf, data, target, cv=10)
print np.mean(recall)
