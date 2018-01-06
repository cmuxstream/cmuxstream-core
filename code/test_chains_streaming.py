#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use("Agg")
from Chains import Chains
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.neighbors import NearestNeighbors
import sys

if __name__ == "__main__":
    data = loadmat("../data/synDataNoisy.mat")
    X = data["X"]
    y = data["y"].ravel()
    anomalies = y == 1

    Xnoisy = X

    print "Testing chain add/remove...",
    k = 50
    nchains = 5
    depth = 2
    print k, nchains, depth
    cf = Chains(k=k, nchains=nchains, depth=depth, projections='streamhash')

    print "\tFitting..."
    cf.fit(Xnoisy)
    scores1 = cf.bincount(Xnoisy)

    print "\tRemoving...",
    for x in Xnoisy:
        cf.update(x.reshape(1,-1), action='remove')
    scores2 = cf.bincount(Xnoisy)
    print np.allclose(scores2, np.zeros((Xnoisy.shape[0],
                                         depth)))

    print "\tAdding...",
    for x in Xnoisy:
        cf.update(x.reshape(1,-1), action='add')
    scores3 = cf.bincount(Xnoisy)
    print np.allclose(scores3, scores1)

    sys.exit()

    print "Stability..."
    ks = [50, 100, 100]
    ncs = [50, 50, 100]
    depth = 10
    for k, nchains in zip(ks, ncs):
        print k, nchains, depth
        aps = []
        for trial in range(5):
            print "\tTrial", trial,
            cf = Chains(k=k, nchains=nchains, depth=depth, projections='streamhash')
            print "fit",
            cf.fit(Xnoisy)
            print "score",
            s = -cf.score(Xnoisy)
            average_precision = average_precision_score(y, s)
            print "AP:", average_precision
            aps.append(average_precision)
        print "Mean AP:", np.mean(aps), "std:", np.std(aps)
