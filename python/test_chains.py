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

    print "Chains...",
    k = 50
    nchains = 50
    depth = 10
    print k, nchains, depth
    depths = range(depth)
    bincounts = np.zeros((Xnoisy.shape[0], depth), dtype=np.float)
    cf = Chains(k=k, nchains=nchains, depth=depth, projections='streamhash')
    cf.fit(Xnoisy)
    bincounts = cf.bincount(Xnoisy)

    # bincount histogram
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

    # bincount pr
    f, ax = plt.subplots()
    print "Bincount PR:"
    for i, d in enumerate(np.arange(0, 10, 1)):
        s = -bincounts[:,d]
        precision, recall, _ = precision_recall_curve(y, s, pos_label=1)
        average_precision = average_precision_score(y, s)
        print "\td:", d, "AP:", average_precision
        plt.plot(recall, precision, label="d=" + str(d) + " AP=" +
                 '{:.3f}'.format(average_precision))
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='lower left')
    plt.savefig("chains_bincount_pr_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")

    # lociscore histogram
    f, ax = plt.subplots()
    lociscores = cf.lociscore(Xnoisy)
    ax.boxplot([lociscores[anomalies,d] for d in depths],
               positions=np.arange(0.0, 2.0*len(depths), 2.0),
               boxprops={'color': '#e41a1c'}, sym='', whis=[10,90],
               medianprops={'color': '#e41a1c'})
    ax.boxplot([lociscores[~anomalies,d] for d in depths],
               positions=np.arange(1.0, 2.0*len(depths)+1, 2.0),
               boxprops={'color': '#377eb8'}, sym='', whis=[10,90],
               medianprops={'color': '#377eb8'})
    plt.xlim(-1, 2*len(depths))
    plt.xticks(np.arange(0.0, 2.0*len(depths), 2.0), depths, rotation=90)
    plt.grid()
    plt.xlabel(r"Depth $d$")
    plt.ylabel(r"LOCI score")
    plt.savefig("chains_lociscore_hist_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")

    # lociscore pr
    print "LOCI PR:"
    f, ax = plt.subplots()
    for i, d in enumerate(np.arange(0, 10, 1)):
        s = -lociscores[:,d]
        precision, recall, _ = precision_recall_curve(y, s, pos_label=1)
        average_precision = average_precision_score(y, s)
        print "\td:", d, "AP:", average_precision
        plt.plot(recall, precision, label="d=" + str(d) + " AP=" +
                 '{:.3f}'.format(average_precision))
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='lower left')
    plt.savefig("chains_lociscore_pr_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")

    # anomaly score histogram
    f, ax = plt.subplots()
    anomalyscores = cf.score(Xnoisy)
    ax.boxplot(anomalyscores[anomalies],
               positions=[0.0],
               boxprops={'color': '#e41a1c'}, sym='', whis=[10,90],
               medianprops={'color': '#e41a1c'})
    ax.boxplot(anomalyscores[~anomalies],
               positions=[1.0],
               boxprops={'color': '#377eb8'}, sym='', whis=[10,90],
               medianprops={'color': '#377eb8'})
    plt.xlim(-1, 2)
    plt.xticks([])
    plt.grid()
    plt.xlabel("")
    plt.ylabel(r"Anonomaly score")
    plt.savefig("chains_anomalyscore_hist_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")

    # anomaly score pr
    print "Anomaly score PR:"
    f, ax = plt.subplots()
    s = -anomalyscores
    precision, recall, _ = precision_recall_curve(y, s, pos_label=1)
    average_precision = average_precision_score(y, s)
    print "\tAP:", average_precision
    plt.plot(recall, precision, label=" AP=" +
             '{:.3f}'.format(average_precision))
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='lower left')
    plt.savefig("chains_anomalyscore_pr_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")

    # unique bin id's
    print "Unique bins..."
    nbins = []
    for d in range(depth):
        nbins_d = []
        for c in cf.chains:
            n = len(c.cmsketches[d].keys())
            nbins_d.append(n)
        nbins.append(nbins_d)
    f, ax = plt.subplots()
    ax.boxplot(nbins, boxprops={'color': 'k'}, sym='', whis=[10,90],
               medianprops={'color': 'k'})
    plt.grid()
    plt.xlabel(r"Depth $d$")
    plt.ylabel(r"No. of unique bin IDs")
    plt.savefig("chains_binids_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")

    print "Stability..."
    ks = [50, 100, 100]
    ncs = [50, 50, 100]
    depth = 10
    for k, nchains in zip(ks, ncs):
        print k, nchains, depth
        depths = range(depth)
        aps = []
        for trial in range(5):
            print "\tTrial", trial
            cf = Chains(k=k, nchains=nchains, depth=depth, projections='streamhash')
            cf.fit(Xnoisy)
            s = -cf.score(Xnoisy)
            average_precision = average_precision_score(y, s)
            aps.append(average_precision)
        print "Mean AP:", np.mean(aps), "std:", np.std(aps)
