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
    data = loadmat("../data/synData.mat")
    X = data["X"]
    y = data["y"].ravel()
    anomalies = y == 1

    Xnoisy = np.concatenate((X, np.random.normal(loc=0.5, scale=0.05,
                                                 size=(X.shape[0], 100))), axis=1)

    print "Chains..."
    k = 50
    nchains = 20
    depth = 10
    depths = range(depth)
    bincounts = np.zeros((Xnoisy.shape[0], depth), dtype=np.float)
    cf = Chains(k=k, nchains=nchains, depth=depth)
    cf.fit(Xnoisy)
    bincounts = cf.bincount(Xnoisy)

    f, ax = plt.subplots()
    ax.boxplot([bincounts[anomalies,d] for d in depths],
               positions=np.arange(0.0, 2.0*len(depths), 2.0),
               boxprops={'color': '#e41a1c'}, sym='', whis=[10,90],
               medianprops={'color': '#e41a1c'})
    ax.boxplot([bincounts[~anomalies,d] for d in depths],
               positions=np.arange(1.0, 2.0*len(depths)+1, 2.0),
               boxprops={'color': '#377eb8'}, sym='', whis=[10,90],
               medianprops={'color': '#377eb8'})
    plt.xlim(-1, 2*len(depths))
    plt.xticks(np.arange(0.0, 2.0*len(depths), 2.0), depths, rotation=90)
    plt.grid()
    plt.xlabel(r"Depth $d$")
    plt.ylabel(r"No. of neighbors")
    plt.savefig("chains_bincount_hist_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")

    f, ax = plt.subplots()
    for i, d in enumerate(np.arange(0, 10, 1)):
        s = -bincounts[:,d]
        precision, recall, _ = precision_recall_curve(y, s, pos_label=1)
        average_precision = average_precision_score(y, s)
        print "d:", d, "AP:", average_precision
        plt.plot(recall, precision, lw=1, label="d=" + str(d) + " AP=" +
                 '{:.3f}'.format(average_precision))
    plt.grid()
    plt.title("Scores = Bin-counts")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='lower left')
    plt.savefig("chains_bincount_pr_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")
