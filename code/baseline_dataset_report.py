#!/usr/bin/env python

from constants import *
from Chains import Chains
import numpy as np
import glob
from sklearn.metrics import average_precision_score, roc_auc_score
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

    scores = []
    aps = []
    aucs = []
    with open("results_" + filename.split("/")[-1], "r") as f:
        for line in f:
            line = line.strip()
            s = map(float, line.split(","))
            scores.append(s)
            ap = average_precision_score(y, s)
            auc = roc_auc_score(y, s) 
            aps.append(ap)
            aucs.append(auc)
    scores = np.array(scores)
    aps = np.array(aps)
    aucs = np.array(aucs)

    mean_ap = np.mean(aps)
    std_ap = np.std(aps)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    filename = filename.split("/")[-1]
    print filename,
    print "\tAP:", "{:.4f}".format(mean_ap),
    print "\pm", "{:.4f}".format(std_ap),
    print "\tAUC:", "{:.4f}".format(mean_auc),
    print "\pm", "{:.4f}".format(std_auc)
