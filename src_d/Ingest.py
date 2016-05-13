"""
Remove formatting from training data 
"""
def custom_strip(string):
    s = string.strip("\n")
    s = s.strip("\r")
    return s

"""
Converts a .csv of training data into a list of dicts
"""
def ingest_training_data(training_data_file):
    instances = []
    classes   = []
    with open(training_data_file) as training_data:
        feature_string = training_data.next()
        features       = feature_string.split(",")[:-1]
        labels         = list(features)
        labels.append("class")
        ### Get a row, make a dict
        ### Keys are feature names, values are feature values
        for row in training_data:
            values   = [ custom_strip(x) for x in row.split(",") ]
            klass    = values[-1]
            if klass not in classes:
                classes.append(klass)
            instance = { l:v for l,v in zip(labels, values) }
            instances.append(instance)
            
        return instances, features, classes





