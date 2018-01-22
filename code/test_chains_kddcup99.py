#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Chains import Chains
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import sys
from tqdm import tqdm

if __name__ == "__main__":
    print "Reading data..."
    data = np.loadtxt("../data/http_smtp_continuous.csv.gz",
                      delimiter=",")
    X = data[:,:-1]
    y = data[:,-1].astype(int)

    print "Chains...",
    k = 100
    nchains = 100
    depth = 10
    print k, nchains, depth

    cf = Chains(k=k, nchains=nchains, depth=depth, projections='streamhash')

    nelements = X.shape[0]
    initial_sample_size = 181877 #int(1.0/100 * nelements)
    cf.fit(X[:initial_sample_size,:])

    with open("kdd99_scores_k" + str(k) + "_C" + str(nchains) +
              "_d" + str(depth) + ".txt", "w") as f:
        for idx in tqdm(range(initial_sample_size, X.shape[0]), desc="Streaming..."):
            x = X[idx,:]
            cf.update(x.reshape(1,-1), action="add")

            if (idx % 10000) == 0:
                s = -cf.score(X[:idx+1,:])
                average_precision = average_precision_score(y[:idx+1], s)
                f.write(str(idx) + "\t" + "{:.4f}".format(average_precision) + "\t\n")
                f.flush()

        s = -cf.score(X)
        average_precision = average_precision_score(y, s)
        f.write(str(idx) + "\t" + "{:.4f}".format(average_precision) + "\t\n")
        f.flush()
