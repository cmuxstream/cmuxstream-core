#!/usr/bin/env python

import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

class LODA:
    def __init__(self, nhistograms=50, nbins=30):
        self.nhistograms = nhistograms
        self.histograms = []
        self.nbins = nbins
        self.projector = None

    def fit(self, X):
        self.projector = SparseRandomProjection(n_components=self.nhistograms,
                                                density=1/3.0)
        Z = self.projector.fit_transform(X) 
        for i in range(self.nhistograms):
            counts, breaks = np.histogram(Z[:,i], bins=self.nbins, density=False)
            self.histograms.append((counts, breaks))

    def score(self, X):
        scores = np.zeros((X.shape[0], self.nhistograms))
        Z = self.projector.transform(X) 
        for i in range(self.nhistograms):
            counts, breaks = self.histograms[i]
            for j in range(Z.shape[0]):
                if Z[j,i] < breaks[0]:
                    bin = 0
                elif Z[j,i] > breaks[len(breaks)-1]:
                    bin = len(breaks) - 1
                else:
                    bin = min(np.trunc((Z[j,i] - breaks[0])/(breaks[1] - breaks[0])),
                              self.nbins-1)
                bin = int(bin)
                scores[j,i] = counts[bin] 
        return -np.mean(np.log(scores), axis=1)
