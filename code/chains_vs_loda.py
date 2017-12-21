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

cwd = os.getcwd()
sys.path.insert(0, cwd + '/../baselines/LODA/')
import loda, ensemble_support

if __name__ == "__main__":
    data = loadmat("../data/synDataNoisy.mat")
    X = data["X"]
    y = data["y"].ravel()
    anomalies = y == 1

    # chains
    k = 25
    nchains = 10
    depth = 10
    print "Chains:", k, nchains, depth

    cf = Chains(k=k, nchains=nchains, depth=depth)
    cf.fit(X)
    print "Scoring..."
    anomalyscores = -cf.score(X)

    # loda
    print "LODA..."
    lodares = loda.loda(X, sparsity=0.3, mink=1, maxk=25)
    model = ensemble_support.generate_model_from_loda_result(lodares, X, y)
    lodascores = model.anom_score

    f, ax = plt.subplots()

    # chains
    precision, recall, _ = precision_recall_curve(y, anomalyscores, pos_label=1)
    average_precision = average_precision_score(y, anomalyscores)
    plt.plot(recall, precision, label="Chains AP=" +
             '{:.3f}'.format(average_precision))
    print "\tChains AP:", average_precision

    # LODA
    precision, recall, _ = precision_recall_curve(y, lodascores, pos_label=1)
    average_precision = average_precision_score(y, lodascores)
    plt.plot(recall, precision, label="LODA AP=" +
             '{:.3f}'.format(average_precision))
    print "\tLODA AP:", average_precision

    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='lower left')
    plt.savefig("chains_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + "_vs_loda" + ".pdf",
                bbox_inches="tight")
