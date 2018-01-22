#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import sys

if __name__ == "__main__":
    data_file = sys.argv[1]
    scores_file = sys.argv[2]
    print >> sys.stderr, "AP over time.",
    print >> sys.stderr, "Data file:", data_file 
    print >> sys.stderr, "Scores file:", scores_file 

    X = np.loadtxt(data_file, delimiter="\t")
    y = X[:,-1].astype(int)

    with open(scores_file, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            nrows = int(fields[0])

            if np.sum(y[:nrows]) == 0:
                continue

            scores = map(float, fields[1].split(" ")) 
            scores = -np.array(scores)
            ap = average_precision_score(y[:nrows], scores[:nrows]) 

            print nrows, ap
