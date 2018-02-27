#!/usr/bin/env python

import itertools
from StreamhashProjection import StreamhashProjection
import numpy as np
import tqdm
tqdm.tqdm.monitor_interval = 0

class Chain:

    def __init__(self, deltamax, depth=25):
        k = len(deltamax)
        self.deltamax = deltamax # feature ranges
        self.depth = depth
        self.fs = [np.random.randint(0, k) for d in range(depth)]
        self.cmsketches = [None] * depth
        self.shift = np.random.rand(k) * deltamax

    def fit(self, X, verbose=False, update=False):
        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            if update:
                cmsketch = self.cmsketches[depth]
            else:
                cmsketch = {}
            for prebin in prebins:
                l = tuple(np.floor(prebin).astype(np.int))
                if not l in cmsketch:
                    cmsketch[l] = 0
                cmsketch[l] += 1
            self.cmsketches[depth] = cmsketch
        return self

    def bincount(self, X):
        scores = np.zeros((X.shape[0], self.depth))
        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
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

    def score(self, X, adjusted=False):
        # scale score logarithmically to avoid overflow:
        #    score = min_d [ log2(bincount x 2^d) = log2(bincount) + d ]
        scores = self.bincount(X)
        depths = np.array([d for d in range(1, self.depth+1)])
        scores = np.log2(1.0 + scores) + depths # add 1 to avoid log(0)
        return np.min(scores, axis=1)

class Chains:
    def __init__(self, k=50, nchains=100, depth=25, seed=42):
        self.nchains = nchains
        self.depth = depth
        self.chains = []
        self.projector = StreamhashProjection(n_components=k,
                                              density=1/3.0,
                                              random_state=seed)

    def fit(self, X):
        projected_X = self.projector.fit_transform(X)
        deltamax = np.ptp(projected_X, axis=0)/2.0
        deltamax[deltamax==0] = 1.0
        for i in tqdm.tqdm(range(self.nchains), desc='Fitting...'):
            c = Chain(deltamax, depth=self.depth)
            c.fit(projected_X)
            self.chains.append(c)

    def score(self, X, adjusted=False):
        projected_X = self.projector.transform(X)
        scores = np.zeros(X.shape[0])
        for i in tqdm.tqdm(range(self.nchains), desc='Scoring...'):
            chain = self.chains[i]
            scores += chain.score(projected_X, adjusted)
        scores /= float(self.nchains)
        return scores
