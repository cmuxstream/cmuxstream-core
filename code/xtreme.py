#!/usr/bin/env python

from constants import *
import numpy as np
from sklearn.random_projection import SparseRandomProjection

class Xtreme:
    projected_X = None
    projector = None
    cmsketch = {}

    def __init__(self, binwidth=1.0):
        # n_components >= 4 log(n_samples) / (eps^2 / 2 - eps^3 / 3)
        # See Theorem 2.1:
        #   http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.3654
        #
        # For eps = 0.1, n_components >= n_features
        self.projector = SparseRandomProjection(n_components=10,
                                                density=1/3.0,
                                                random_state=SEED)
        self.binwidth = binwidth

    def fit(self, X):
        projected_X = self.projector.fit_transform(X)
        binned_X = np.floor(projected_X/self.binwidth).astype(np.int)
        for row in binned_X:
            l = tuple(row)
            if not l in self.cmsketch:
                self.cmsketch[l] = 1
            self.cmsketch[l] += 1

    def predict(self, X, threshold=0.5):
        """ 1 if outlier, 0 otherwise """
        ypred = np.ones(X.shape[0], dtype=np.int)
        yscore = np.zeros(X.shape[0], dtype=np.float)

        assert self.cmsketch is not None

        projected_X = self.projector.fit_transform(X)
        binned_X = np.floor(projected_X/self.binwidth).astype(np.int)
        for i, row in enumerate(binned_X):
            l = tuple(row)
            bincount = 0
            if l in self.cmsketch:
                bincount = self.cmsketch[l]
            yscore[i] = float(bincount)

        return ypred, yscore
