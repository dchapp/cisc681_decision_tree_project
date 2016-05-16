import numpy as np
import NestedDictDT as DecisionTree
import printTree
import kFold

def LoadMPGData():
    with open('./data/mpg_cars.csv') as f:
        data = f.readlines()
    data = map(lambda x: x.rstrip().split(','), data)
    data = np.array(data)
    data = np.hstack((data[:,1:], data[:,0:1])) # make class the last column
    return data

if __name__ == '__main__':
    data = LoadMPGData()
    model = DecisionTree(header=True, continuous='dynamic', min_leaf=2, bins=3)
    accuracy = kFold(data, model, 10)
    print sum(accuracy)/len(accuracy)

    #model.fit(data)
    #predict = model.predict(test)
    #print 'Accuracy:',
    #print float(np.sum(true == predict))/len(predict)
    #tree = model.getTree()
    #printTree(tree)
    #pprint(tree)

