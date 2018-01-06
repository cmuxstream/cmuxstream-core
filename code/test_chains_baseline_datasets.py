#!/usr/bin/env python

from constants import *
from Chains import Chains
import glob
import numpy as np
from sklearn.metrics import average_precision_score
import sys

def read_dataset(filename):
    data = np.loadtxt(filename, delimiter=',')
    n,m = data.shape
    X = data[:,0:m-1]
    y = data[:,m-1]
    filename = filename.split("/")[-1]
    print "Dataset:", filename, X.shape, y.shape
    return X, y

if __name__ == "__main__":
    filename = sys.argv[1]
    X, y = read_dataset(filename)
    nruns = int(sys.argv[2])

    times = np.zeros(nruns, dtype=float)
    aps = np.zeros(nruns, dtype=float)
    aucs = np.zeros(nruns, dtype=float)
    scores = []

    print "Chains...",
    k = 50
    nchains = 100
    depth = 10
    print k, nchains, depth

    for run in range(nruns):
        print "\tRun", run,
        cf = Chains(k=k, nchains=nchains, depth=depth)
        cf.fit(X)
        anomalyscores = -cf.score(X)
        scores.append(anomalyscores)
        ap = average_precision_score(y, anomalyscores) 
        print "AP =", ap

    results_filename = "results_" + filename.split("/")[-1]
    with open(results_filename, "w") as f:
        for s in scores:
            f.write(",".join(["{:.12f}".format(x) for x in s]))
            f.write("\n")
