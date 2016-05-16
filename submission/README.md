## University of Delaware CISC-481/681 Project 3: Decision Trees
This project contains our team's implementations of decision trees in Python using the ID3 algorith.
We evaluate our decision trees against two datasets from the UCI machine learning repository:
1. The Auto-MPG dataset (MPG)
2. The Wisconsin Breast Cancer dataset (WBC)

In both cases, we use a discretization policy to deal with continuous-valued data.

### Dependencies
Python 2.7
numpy
argparse
pydot

### Usage
The decision tree is built by invoking the driver.py script with appropriate arguments.

To view help, run:
```
$python driver.py -h
```

To train against the MPG dataset and test, run:
```
$python driver.py --train ../test/cars/train.csv --test ../test/cars/test.csv -p Good -r
```

To train against the WBC dataset and test, run:
```
$python driver.py --train ../test/WBC/wdbc-train-csv.data --test ../test/WBC/wdbc-test-csv.data -p 2 -r
```

To enable k-fold cross validation, append the -k flag to any of the above commands. 


