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
    orig_dists = []
    projected_dists = []
    with open("../code-cpp/distances_k100.txt", "r") as f:
        for line in f:
            true, approx = map(float, line.strip().split(" "))
            orig_dists.append(true)
            projected_dists.append(approx)

    orig_dists = np.array(orig_dists)
    projected_dists = np.array(projected_dists)
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
    plt.savefig("cpp_distances_euclidean_k100.pdf", bbox_inches="tight")
