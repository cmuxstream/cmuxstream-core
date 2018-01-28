#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use("Agg")
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

    bincounts = []
    lociscores = []
    anomalyscores = []
    filename = sys.argv[1]
    with open(filename, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            anomalyscore = float(fields[0])
            bincount = map(float, fields[1].split(" "))
            lociscore = map(float, fields[2].split(" "))
            bincounts.append(bincount)
            lociscores.append(lociscore)
            anomalyscores.append(anomalyscore)
    bincounts = np.array(bincounts)
    lociscores = np.array(lociscores)
    anomalyscores = -np.array(anomalyscores)

    depth = bincounts.shape[1]
    multiplier = np.array([2.0 ** d for d in range(1, depth+1)])

    bincounts *= multiplier

    medians = np.zeros((len(CLASSES), depth))
    pct5s = np.zeros((len(CLASSES), depth))
    pct95s = np.zeros((len(CLASSES), depth))

    # bincount histogram
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
        plt.errorbar(x=xs, y=ms, yerr=yerr[::-1], label=str(idx), alpha=0.75) 
    plt.grid()
    plt.legend(ncol=6, bbox_to_anchor=(1.05,1.15))
    plt.xlabel(r"Depth $d$")
    plt.ylabel(r"Bin count $\times 2^d$")
    plt.yscale("symlog", basey=2.0)
    plt.savefig("chainscpp_type_bincounts_d" + str(depth) + ".pdf",
                bbox_inches="tight")

    # lociscore histogram
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
        plt.errorbar(x=xs, y=ms, yerr=yerr[::-1], label=str(idx), alpha=0.75) 

    plt.grid()
    plt.legend(ncol=6, bbox_to_anchor=(1.05,1.15))
    plt.xlabel(r"Depth $d$")
    plt.ylabel(r"LOCI score$")
    plt.yscale("symlog", basey=2.0)
    plt.savefig("chainscpp_type_lociscores_d" + str(depth) + ".pdf",
                bbox_inches="tight")

    # new anomaly scores
    # bincount[i][d] = bc[i] * 2^d
    # anomaly score = bincount at max d as long as unscaled bincount > sizelim
    #unscaled_bincounts = bincounts / multiplier
    #anomalyscores = -np.mean(bincounts, axis=1)

    # anomaly score histogram
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
        plt.errorbar(x=[idx], y=[ms], yerr=yerr[::-1], label=str(idx), alpha=0.75,
                     lw=3) 

    plt.grid()
    plt.legend(ncol=6, bbox_to_anchor=(1.05,1.15))
    plt.xlabel(r"Point type")
    plt.ylabel(r"anomaly score")
    plt.savefig("chainscpp_type_anomalyscores_d" + str(depth) + ".pdf",
                bbox_inches="tight")

    average_precision = average_precision_score(y, anomalyscores)
    print "AP:", average_precision
