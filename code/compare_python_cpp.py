#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use("Agg")
from Chains import Chains
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import precision_recall_curve, average_precision_score
import sys

if __name__ == "__main__":
    data = loadmat("../data/synDataNoisy.mat")
    X = data["X"]
    y = data["y"].ravel()
    Xnoisy = X

    print "Chains-py...",
    k = 10
    nchains = 10
    depth = 10
    print k, nchains, depth
    depths = range(depth)
    cf = Chains(k=k, nchains=nchains, depth=depth)
    cf.fit(Xnoisy)

    deltamax = np.zeros(k, dtype=np.float)
    for c in cf.chains:
        deltamax += c.deltamax
    deltamax /= nchains
    print "avg deltama", np.mean(deltamax)

    medians = np.zeros((len(CLASSES), depth))
    pct5s = np.zeros((len(CLASSES), depth))
    pct95s = np.zeros((len(CLASSES), depth))

    # bincount histogram
    bincounts = cf.bincount(Xnoisy)
    print bincounts[0,:]
    for idx, c in enumerate(CLASSES):
        for d in range(depth):
            s = bincounts[c,d]
            medians[idx,d] = np.percentile(s, q=50.0)
            pct5s[idx,d] = np.percentile(s, q=5.0)
            pct95s[idx,d] = np.percentile(s, q=95.0)

    plt.figure()
    xs = np.arange(depth) + 1.0
    for idx in range(len(CLASSES)):
        ms = medians[idx,:]
        yerr = [pct95s[idx,:] - ms, ms - pct5s[idx,:]]
        plt.errorbar(x=xs, y=ms, yerr=yerr, label=str(idx), alpha=0.75) 
    plt.grid()
    plt.legend(ncol=6, bbox_to_anchor=(1.05,1.15))
    plt.xlabel(r"Depth $d$")
    plt.ylabel(r"Bin count")
    plt.savefig("python_bincounts_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")

    # cpp bincount histogram
    bincounts = []
    with open("../code-cpp/synDataBincounts.txt", "r") as f:
        for line in f:
            bincounts.append(map(float, line.strip().split(" ")))
    bincounts = np.array(bincounts)
    for idx, c in enumerate(CLASSES):
        for d in range(depth):
            s = bincounts[c,d]
            medians[idx,d] = np.percentile(s, q=50.0)
            pct5s[idx,d] = np.percentile(s, q=5.0)
            pct95s[idx,d] = np.percentile(s, q=95.0)

    plt.figure()
    xs = np.arange(depth) + 1.0
    for idx in range(len(CLASSES)):
        ms = medians[idx,:]
        yerr = [pct95s[idx,:] - ms, ms - pct5s[idx,:]]
        plt.errorbar(x=xs, y=ms, yerr=yerr, label=str(idx), alpha=0.75) 
    plt.grid()
    plt.legend(ncol=6, bbox_to_anchor=(1.05,1.15))
    plt.xlabel(r"Depth $d$")
    plt.ylabel(r"Bin count")
    plt.savefig("cpp_bincounts_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")

    sys.exit()

    # lociscore histogram
    lociscores = cf.lociscore(Xnoisy)
    multiplier = np.array([2.0 ** d for d in range(1, depth+1)])
    lociscores /= multiplier

    for idx, c in enumerate(CLASSES):
        for d in range(depth):
            s = lociscores[c,d]
            medians[idx,d] = np.percentile(s, q=50.0)
            pct5s[idx,d] = np.percentile(s, q=5.0)
            pct95s[idx,d] = np.percentile(s, q=95.0)

    plt.figure()
    xs = np.arange(depth) + 1.0
    for idx in range(len(CLASSES)):
        ms = medians[idx,:]
        yerr = [pct95s[idx,:] - ms, ms - pct5s[idx,:]]
        plt.errorbar(x=xs, y=ms, yerr=yerr, label=str(idx), alpha=0.75) 

    plt.grid()
    plt.legend(ncol=6, bbox_to_anchor=(1.05,1.15))
    plt.xlabel(r"Depth $d$")
    plt.ylabel(r"LOCI score / $2^d$")
    plt.savefig("python_lociscores_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")
    
    # anomaly score histogram
    anomalyscores = -cf.score(Xnoisy)

    medians = np.zeros(len(CLASSES))
    pct5s = np.zeros(len(CLASSES))
    pct95s = np.zeros(len(CLASSES))
    for idx, c in enumerate(CLASSES):
        s = anomalyscores[c]
        medians[idx] = np.percentile(s, q=50.0)
        pct5s[idx] = np.percentile(s, q=5.0)
        pct95s[idx] = np.percentile(s, q=95.0)

    plt.figure()
    xs = np.arange(depth) + 1.0
    for idx in range(len(CLASSES)):
        ms = medians[idx]
        yerr = [[pct95s[idx] - ms], [ms - pct5s[idx]]]
        plt.errorbar(x=[idx], y=[ms], yerr=yerr, label=str(idx), alpha=0.75,
                     lw=3) 

    plt.grid()
    plt.legend(ncol=6, bbox_to_anchor=(1.05,1.15))
    plt.xlabel(r"Point type")
    plt.ylabel(r"anomaly score")
    plt.savefig("python_anomscores_k" + str(k) +
                "c" + str(nchains) + "d" + str(depth) + ".pdf",
                bbox_inches="tight")

    average_precision = average_precision_score(y, anomalyscores)
    print "AP:", average_precision
