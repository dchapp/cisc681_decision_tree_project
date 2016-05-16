from Evaluation import recall
from NestedDictDT import DecisionTree
from CrossValidation import kFold
from PrintTree import treeInfo, printTree
import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, required=True,
        help='file with training data')
parser.add_argument('--test', type=str, required=True,
        help='file with testing data')
parser.add_argument('-r', '--header', help='if file has headers (labels)',
        action='store_true')
parser.add_argument('-c', '--cont', type=str, required=False,
        default='static', help='Discretization type: false, static, or dynamic')
parser.add_argument('-b', '--bins', type=int, required=False,
        default=3, help='number of bins to discretize data')
parser.add_argument('-k', '--kfold', help='perform crossvalidation?',
        action='store_true')
parser.add_argument('-l', '--leaf', help='minleaf', default=2)
parser.add_argument('-d', '--depth', help='maxdepth', default=6)
parser.add_argument('-p', '--positive', help='positive class label', required=True)
args = parser.parse_args()

if args.cont == 'false':
    args.cont = False

with open(args.train) as train_f:
    train_data = train_f.readlines()
train_data = map(lambda x: x.rstrip().split(','), train_data)
train_data = np.array(train_data)
train_data = np.hstack((train_data[:,1:], train_data[:,0:1]))

with open(args.test) as test_f:
    test_data = test_f.readlines()
test_data = map(lambda x: x.rstrip().split(','), test_data)
test_data = np.array(test_data)
test_data = np.hstack((test_data[:,1:], test_data[:,0:1]))

model = DecisionTree(header=args.header, continuous=args.cont,
        min_leaf=args.leaf, bins=args.bins, max_depth=args.depth)

if args.kfold:
    accuracy = kFold(model, train_data, 10, args.positive)
    print 'kFold results:',
    print accuracy
else:
    true = np.array(test_data[1:,-1])
    test_data = np.array(test_data[:,:-1])

    model.fit(train_data)
    predict = model.predict(test_data)

    print 'Recall:',
    print recall(true, predict, args.positive)

    tree = model.getTree()
    depth, node_count = treeInfo(tree)
    print 'Max Depth: %d, Nodes: %d' % (depth, node_count)

    printTree(tree)
