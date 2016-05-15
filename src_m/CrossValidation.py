import numpy as np

def kFold(data, model, k):
    if len(data) < k:
        print 'Error: k must be smaller than the number of data points'

    # Get data without header
    header = model.header
    data = data[header:]

    model.header = False # Don't care about feature names for crossval

    # Assign data to folds - makes sure at least one data point in each fold
    unique = []
    while len(unique) != k:
        assign = np.random.randint(k, size=len(data))
        unique = np.unique(assign)

    accuracy = []

    # Fit and predict with folds
    for i in range(k):
        data_train = np.array(data[assign!=i])
        data_test = np.array(data[assign==i])
        data_true = np.array(data_test[:,-1])
        data_test = np.array(data_test[:,:-1])

        model.fit(data_train)
        data_predict = model.predict(data_test)

        # Append accuracy of current prediction
        accuracy.append(float(np.sum(data_true==data_predict))/len(data_true))

    # Return list of accuracies
    return accuracy
