#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from sklearn.random_projection import SparseRandomProjection
from StreamhashProjection import StreamhashProjection
import sys
import time

if __name__ == "__main__":
    metric = sys.argv[1]

    data = loadmat("../data/synDataNoisy.mat")
    X = data["X"]
    y = data["y"].ravel()
    Xnoisy = X

    orig_dists = pdist(Xnoisy, metric=metric).ravel()
    pd = []
    shd = []
    ks = [50, 100, 1000]
    for i, k in enumerate(ks):
        print "k:", k, metric
        print "SparseRandomProjection...",
        t0 = time.time()
        projector = SparseRandomProjection(n_components=k,
                                           density=1/3.0,
                                           random_state=SEED)
        projected_X = projector.fit_transform(Xnoisy)
        print "done in", time.time() - t0, "s"

        projected_dists = pdist(projected_X, metric=metric).ravel()
        nonzero = orig_dists != 0

        f, ax = plt.subplots()
        plt.hexbin(orig_dists[nonzero], projected_dists[nonzero],
                   gridsize=100, cmap=plt.cm.Blues, bins='log')
        plt.xlabel("Original distances")
        plt.ylabel("Projected distances")
        cb = plt.colorbar()
        cb.set_label("log10(count)")
        plt.grid()
        ax.set_aspect("equal")
        plt.savefig("sparse_projection_" + metric + "_k"
                    + str(k) + ".pdf", bbox_inches="tight")

        print "Streamhash...", metric,
        t0 = time.time()

        projector = StreamhashProjection(n_components=k,
                                         density=1/3.0,
                                         random_state=SEED)

        projected_X = projector.fit_transform(Xnoisy)
        print "done in", time.time() - t0, "s"

        projected_dists = pdist(projected_X, metric=metric).ravel()
        nonzero = orig_dists != 0

        f, ax = plt.subplots()
        plt.hexbin(orig_dists[nonzero], projected_dists[nonzero],
                   gridsize=100, cmap=plt.cm.Blues, bins='log')
        plt.xlabel("Original distances")
        plt.ylabel("Projected distances")
        cb = plt.colorbar()
        cb.set_label("log10(count)")
        plt.grid()
        plt.savefig("streamhash_projection_" + metric + "_k"
                    + str(k) + ".pdf", bbox_inches="tight")
