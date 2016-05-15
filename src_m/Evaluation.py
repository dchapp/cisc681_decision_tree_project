import numpy as np

def recall(true, predict, positive):
    tp = np.sum((true == predict) * (true == positive))
    tpfn = np.sum(tree == positive)
    return float(tp)/tpfn
