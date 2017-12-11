#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use("Agg")
from LSHForest import LSHForest
from HistogramForest import HistogramForest
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist
from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection
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

    orig_dists = pdist(Xnoisy, metric='euclidean').ravel()
    maxdist = max(orig_dists)

    bnns = []
    anns = []
    print "Exact nearest neighbors..."
    rs = np.arange(2.0, 12.0, 2.0)
    for r in rs:
        print r
        nn = NearestNeighbors(metric='euclidean', n_jobs=-1)
        nn.fit(Xnoisy)

        anomaly_nns = [len(l)
                       for l in nn.radius_neighbors(Xnoisy[anomalies,:],
                                                    radius=r,
                                                    return_distance=False)]
        benign_nns = [len(l)
                       for l in nn.radius_neighbors(Xnoisy[~anomalies,:],
                                                    radius=r,
                                                    return_distance=False)]
        anns.append(anomaly_nns)
        bnns.append(benign_nns)

    f, ax = plt.subplots()
    ax.boxplot(anns, positions=np.arange(0.0, 2.0*len(rs), 2.0),
               boxprops={'color': '#e41a1c'}, sym='', whis=[10,90],
               medianprops={'color': '#e41a1c'})
    ax.boxplot(bnns, positions=np.arange(1.0, 2.0*len(rs)+1, 2.0),
               boxprops={'color': '#377eb8'}, sym='', whis=[10,90],
               medianprops={'color': '#377eb8'})
    plt.xlim(-1, 2*len(rs))
    plt.xticks(np.arange(0.0, 2.0*len(rs), 2.0),
               rs)
    plt.grid()
    plt.xlabel(r"Radius $R$")
    plt.ylabel(r"No. of neighbors $\leq R$")
    plt.savefig("nn_exact.pdf", bbox_inches="tight")

    f, ax = plt.subplots()
    rs = np.arange(2.0, 12.0, 2.0)
    for r in rs:
        nn = NearestNeighbors(metric='euclidean', n_jobs=-1).fit(Xnoisy)
        s = [-float(len(l)) for l in nn.radius_neighbors(Xnoisy,
                                                         radius=r,
                                                         return_distance=False)]
        precision, recall, _ = precision_recall_curve(y, s, pos_label=1)
        average_precision = average_precision_score(y, s)
        print "r:", r, "AP:", average_precision
        plt.plot(recall, precision, lw=1,
                 label="R=" + str(r) + " AP=" + '{:.3f}'.format(average_precision))
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("nn_exact_pr.pdf", bbox_inches="tight")

    """
    print "LSHForest"
    depth = 10
    lf = LSHForest(ntrees=100, depth=depth)
    lf.fit(Xnoisy)
    scores = lf.score(Xnoisy)

    f, ax = plt.subplots()
    ax.boxplot([scores[anomalies,d] for d in range(depth)],
               positions=np.arange(0.0, 2.0*depth, 2.0),
               boxprops={'color': '#e41a1c'}, sym='', whis=[10,90],
               medianprops={'color': '#e41a1c'})
    ax.boxplot([scores[~anomalies,d] for d in range(depth)],
               positions=np.arange(1.0, 2.0*depth+1, 2.0),
               boxprops={'color': '#377eb8'}, sym='', whis=[10,90],
               medianprops={'color': '#377eb8'})
    plt.xlim(-1, 2*depth)
    plt.xticks(np.arange(0.0, 2.0*depth, 2.0), np.arange(depth) + 1)
    plt.grid()
    plt.xlabel(r"Depth $d$")
    plt.ylabel(r"No. of neighbors at depth $d$")
    plt.savefig("nn_approx.png", bbox_inches="tight")

    f, ax = plt.subplots()
    for d in range(1, depth):
        s = -scores[:,d]
        precision, recall, _ = precision_recall_curve(y, s, pos_label=1)
        average_precision = average_precision_score(y, s)
        print "Depth:", d, "AP:", average_precision
        plt.plot(recall, precision, lw=1, label="d=" + str(d))
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("nn_approx_pr.png", bbox_inches="tight")
    """
    
    print "E2LSHForest"
    ws = np.arange(2.0, 12.0, 2.0)
    k = 25
    scores = np.zeros((Xnoisy.shape[0], len(ws)), dtype=np.float)
    for i, w in enumerate(ws):
        print w
        lf = HistogramForest(ntrees=100, w=w, k=k)
        lf.fit(Xnoisy)
        scores[:,i] = lf.score(Xnoisy)

    f, ax = plt.subplots()
    ax.boxplot([scores[anomalies,d] for d in range(len(ws))],
               positions=np.arange(0.0, 2.0*len(ws), 2.0),
               boxprops={'color': '#e41a1c'}, sym='', whis=[10,90],
               medianprops={'color': '#e41a1c'})
    ax.boxplot([scores[~anomalies,d] for d in range(len(ws))],
               positions=np.arange(1.0, 2.0*len(ws)+1, 2.0),
               boxprops={'color': '#377eb8'}, sym='', whis=[10,90],
               medianprops={'color': '#377eb8'})
    plt.xlim(-1, 2*len(ws))
    plt.xticks(np.arange(0.0, 2.0*len(ws), 2.0), ws)
    plt.grid()
    plt.xlabel(r"Bin-width $w$")
    plt.ylabel(r"No. of neighbors")
    plt.savefig("nn_approx_hist_k" + str(k) + ".pdf", bbox_inches="tight")

    f, ax = plt.subplots()
    for i, w in enumerate(ws):
        s = -scores[:,i]
        precision, recall, _ = precision_recall_curve(y, s, pos_label=1)
        average_precision = average_precision_score(y, s)
        print "w:", w, "AP:", average_precision
        plt.plot(recall, precision, lw=1, label="w=" + str(w) + " AP=" +
                 '{:.3f}'.format(average_precision))
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("nn_approx_pr_k" + str(k) + ".pdf", bbox_inches="tight")
