#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use("Agg")
from Chains import Chains
import matplotlib.pyplot as plt
import numpy as np
import os 
from scipy.io import loadmat
from scipy.spatial.distance import pdist
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.neighbors import NearestNeighbors
import sys
import loda

if __name__ == "__main__":
    data = loadmat("../data/synDataNoisy.mat")
    X = data["X"]
    y = data["y"].ravel()
    anomalies = y == 1

    # chains
    chains_ap = []
    loda_ap = []
    for i in range(5):
        print "Trial", i

        # chains
        k = 50
        nchains = 50
        depth = 10
        cf = Chains(k=k, nchains=nchains, depth=depth, projections='gaussian')
        cf.fit(X)
        anomalyscores = -cf.score(X)

        l = loda.LODA(nbins=30, nhistograms=50)
        l.fit(X)
        lodascores = l.score(X) # negative loglikelihood

        average_precision = average_precision_score(y, anomalyscores)
        chains_ap.append(average_precision)
        average_precision = average_precision_score(y, lodascores)
        loda_ap.append(average_precision)

    print "Mean LODA AP:", np.mean(loda_ap), np.std(loda_ap)
    print "Mean chains AP:", np.mean(chains_ap), np.std(chains_ap)

    f, ax = plt.subplots()
    precision, recall, _ = precision_recall_curve(y, anomalyscores, pos_label=1)
    plt.plot(recall, precision, label="Chains AP=" +
             '{:.3f}'.format(average_precision))
    precision, recall, _ = precision_recall_curve(y, lodascores, pos_label=1)
    plt.plot(recall, precision, label="LODA AP=" +
             '{:.3f}'.format(average_precision))

    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='lower left')
    plt.savefig("chains_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + "_vs_loda" + ".pdf",
                bbox_inches="tight")
