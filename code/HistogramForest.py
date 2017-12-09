#!/usr/bin/env python

from constants import *
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection

class Histogram:

    def __init__(self, w, k):
        self.w = w
        self.b = np.random.rand(1) * w
        self.counts = {}
        #self.projector = SparseRandomProjection(n_components=k,
        #                                        density=1/3.0,
        #                                        random_state=SEED)
        self.projector = GaussianRandomProjection(n_components=k,
                                                  random_state=SEED)

    def fit(self, X):
        projected_X = self.projector.fit_transform(X) 
        binned_X = np.floor((projected_X + self.b)/self.w).astype(int)
        tuples = [tuple(x) for x in binned_X]
        for t in tuples:
            if not t in self.counts:
                self.counts[t] = 0
            self.counts[t] += 1

    def score(self, X):
        scores = np.zeros(X.shape[0])
        projected_X = self.projector.transform(X)
        binned_X = np.floor((projected_X + self.b)/self.w).astype(int)
        tuples = [tuple(x) for x in binned_X]
        for i, t in enumerate(tuples):
            if not t in self.counts:
                scores[i] = 0.0
            else:
                scores[i] = self.counts[t]
        return scores

class HistogramForest:

    def __init__(self, w, k, ntrees=10):
        self.ntrees = ntrees
        self.w = w
        self.k = k
        self.trees = []

    def fit(self, X):
        for i in range(self.ntrees):
            tree = Histogram(self.w, self.k)
            tree.fit(X)
            self.trees.append(tree)

    def score(self, X):
        scores = np.zeros((X.shape[0], self.ntrees))
        for i, tree in enumerate(self.trees):
            scores[:,i] = tree.score(X)
        return np.mean(scores, axis=1)
        # return np.max(scores, axis=1)
