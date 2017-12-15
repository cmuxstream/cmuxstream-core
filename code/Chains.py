#!/usr/bin/env python

from constants import *
from sklearn.random_projection import SparseRandomProjection
import numpy as np

class Chain:

    def __init__(self, deltamax, depth=25):
        self.deltamax = deltamax # feature ranges
        self.depth = depth
        self.fs = [None] * depth # feature used at each depth
        self.cmsketches = [None] * depth
        self.shift = np.random.rand(len(deltamax)) * deltamax

    def fit(self, X, verbose=False):
        ndim = X.shape[1]
        nrows = X.shape[0]

        assert len(self.shift) == ndim
        assert len(self.deltamax) == ndim

        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(ndim, dtype=np.int)
        for depth in range(self.depth):
            f = np.random.randint(0, ndim) # random feature
            self.fs[depth] = f
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            cmsketch = {}
            for prebin in prebins:
                l = tuple(np.floor(prebin).astype(np.int))
                if not l in cmsketch:
                    cmsketch[l] = 0
                cmsketch[l] += 1
            self.cmsketches[depth] = cmsketch

    def bincount(self, X):
        scores = np.zeros((X.shape[0], self.depth))

        ndim = X.shape[1]
        nrows = X.shape[0]

        assert len(self.shift) == ndim
        assert len(self.deltamax) == ndim

        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(ndim, dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth] 
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            cmsketch = self.cmsketches[depth]
            for i, prebin in enumerate(prebins):
                l = tuple(np.floor(prebin).astype(np.int))
                if not l in cmsketch:
                    scores[i,depth] = 0.0
                else:
                    scores[i,depth] = cmsketch[l]

        return scores

    def lociscore(self, X):
        scores = self.bincount(X)

        multiplier = np.array([2.0 ** d for d in range(1, self.depth+1)])
        scores *= multiplier

        indices = np.arange(scores.shape[0])

        lociscores = np.zeros(scores.shape, dtype=np.float)
        for i in range(X.shape[0]):
            score_i = scores[i,:]
            mean_score_j = np.mean(scores[indices!=i, :])
            lociscores[i,:] = score_i - mean_score_j

        return lociscores

    def score(self, X):
        lociscores = self.lociscore(X)
        scores = np.max(lociscores, axis=1)
        return scores

class Chains:
    def __init__(self, k=50, nchains=100, depth=25, projections='sparse'):
        self.nchains = nchains
        self.depth = depth
        self.chains = []

        if projections == 'sparse':
            self.projector = SparseRandomProjection(n_components=k,
                                                    density=1/3.0,
                                                    random_state=SEED)
        elif projections == 'gaussian':
            self.projector = GaussianRandomProjection(n_components=k,
                                                      random_state=SEED)
        else:
            raise Exception("Unknown projection type: " + projections)

    def fit(self, X):
        projected_X = self.projector.fit_transform(X)
        deltamax = np.ptp(projected_X, axis=0)/2.0
        for i in range(self.nchains):
            print "Fitting chain", i, "..."
            c = Chain(deltamax, depth=self.depth)
            c.fit(projected_X)
            self.chains.append(c)

    def bincount(self, X):
        projected_X = self.projector.transform(X)
        scores = np.zeros((X.shape[0], self.depth))
        for i, chain in enumerate(self.chains):
            scores += chain.bincount(projected_X)
        scores /= float(self.nchains)
        return scores

    def lociscore(self, X):
        projected_X = self.projector.transform(X)
        scores = np.zeros((X.shape[0], self.depth))
        for i, chain in enumerate(self.chains):
            scores += chain.lociscore(projected_X)
        scores /= float(self.nchains)
        return scores

    def score(self, X):
        projected_X = self.projector.transform(X)
        scores = np.zeros(X.shape[0])
        for i, chain in enumerate(self.chains):
            scores += chain.score(projected_X)
        scores /= float(self.nchains)
        return scores
