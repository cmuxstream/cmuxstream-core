#!/usr/bin/env python

from Chains import Chains
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import IsolationForest
import sys

# synthetic data point classes
CLASSES = [range(0,1000), # sparse benign cluster
           range(1000,3000), # dense benign cluster
           range(3000,3050), # clustered anomalies
           range(3050, 3075), # sparse anomalies
           range(3076,3082), # local anomalies
           range(3075,3076)] # single anomaly

if __name__ == "__main__":
    k = 50
    nchains = 50
    depth = 10

    X = np.loadtxt("synDataNoisy.tsv")
    y = np.array([0] * 3000 + [1] * 82)

    cf = Chains(k=k, nchains=nchains, depth=depth)
    cf.fit(X)
    anomalyscores = -cf.score(X)
    ap = average_precision_score(y, anomalyscores) 
    auc = roc_auc_score(y, anomalyscores)
    print "xstream: AP =", ap, "AUC =", auc

    cf = IsolationForest()
    cf.fit(X)
    anomalyscores = -cf.decision_function(X)
    ap = average_precision_score(y, anomalyscores) 
    auc = roc_auc_score(y, anomalyscores)
    print "iForest: AP =", ap, "AUC =", auc
