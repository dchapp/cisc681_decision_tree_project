import sys
import numpy as np

train_f = open(sys.argv[2], "wb")
test_f = open(sys.argv[3], "wb")

with open(sys.argv[1], "rb") as f:
    data = f.readlines()

test_f.write(data[0])

num_instances = int(float(sys.argv[4]) * len(data))
for i in xrange(num_instances):
    ### Get a random index into data
    idx = np.random.randint(len(data))
    ### Add that instance to test set and remove it from consideration
    test_f.write(data[idx])
    data.remove(data[idx])

test_f.close()

for d in data:
    train_f.write(d)

train_f.close()

