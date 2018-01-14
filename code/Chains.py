#!/usr/bin/env python

from constants import *
import itertools
from sklearn.random_projection import SparseRandomProjection
from StreamhashProjection import StreamhashProjection
import numpy as np
import pathos.pools as pp
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

    def fit(self, X, verbose=False, update=False, action='add'):
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
                if action == 'add':
                    if not l in cmsketch:
                        cmsketch[l] = 0
                    cmsketch[l] += 1
                elif action == 'remove':
                    cmsketch[l] -= 1
                    assert cmsketch[l] >= 0
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

    def generate_neighbor_bins(self, bin_i, fs):
        for deltas in itertools.product([-1,0,1], repeat=len(fs)):
            if not np.any(deltas):
                continue
            neighbor_bin = np.copy(bin_i)
            for f, delta in zip(fs, deltas):
                neighbor_bin[f] += delta
            yield tuple(neighbor_bin)

    def neighborbincount(self, X, bins=False):
        scores = np.zeros((X.shape[0], self.depth))
        bincounts = np.zeros((X.shape[0], self.depth))
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
            fs = list(set(self.fs[:depth+1]))
            for i, prebin in enumerate(prebins):
                bin_i = np.floor(prebin).astype(np.int)
                l = tuple(bin_i)
                if l in cmsketch:
                    bincounts[i,depth] = cmsketch[l]
                else:
                    bincounts[i,depth] = 0.0

                bincount_n = []
                for neighbor_bin in self.generate_neighbor_bins(bin_i, fs):
                    if neighbor_bin in cmsketch:
                        bincount_n.append(cmsketch[neighbor_bin])
                    else:
                        bincount_n.append(0)

                scores[i,depth] = np.median(bincount_n)

        if bins == True:
            return scores, bincounts
        return scores

    def lociscore(self, X):
        scores = self.bincount(X)

        multiplier = np.array([2.0 ** d for d in range(1, self.depth+1)])
        scores *= multiplier

        indices = np.arange(scores.shape[0])

        lociscores = np.zeros(scores.shape, dtype=np.float)
        means = np.mean(scores, axis=0)
        lociscores = scores - means

        return lociscores

    def lociscore_adjusted(self, X):
        neighborbincounts, bincounts = self.neighborbincount(X, bins=True)
        multiplier = np.array([2.0 ** d for d in range(1, self.depth+1)])
        lociscores = (bincounts - neighborbincounts) * multiplier
        return lociscores

    def score(self, X, adjusted=False):
        if not adjusted:
            lociscores = self.lociscore(X)
        else:
            lociscores = self.lociscore_adjusted(X)
        scores = np.min(lociscores, axis=1)
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
        elif projections == 'streamhash':
            self.projector = StreamhashProjection(n_components=k,
                                                  density=1/3.0,
                                                  random_state=SEED)
        else:
            raise Exception("Unknown projection type: " + projections)

    def update(self, X, action='add'):
        projected_X = self.projector.fit_transform(X)
        for i in range(self.nchains):
            c = self.chains[i]
            c.fit(projected_X, update=True, action=action)

    def fit(self, X):
        projected_X = self.projector.fit_transform(X)
        deltamax = np.ptp(projected_X, axis=0)/2.0
        deltamax[deltamax==0] = 1.0
        for i in tqdm.tqdm(range(self.nchains), desc='Fitting...'):
            c = Chain(deltamax, depth=self.depth)
            c.fit(projected_X)
            self.chains.append(c)

    def bincount(self, X):
        projected_X = self.projector.transform(X)
        scores = np.zeros((X.shape[0], self.depth))
        for i in tqdm.tqdm(range(self.nchains), desc='Bincount...'):
            chain = self.chains[i]
            scores += chain.bincount(projected_X)
        scores /= float(self.nchains)
        return scores

    def neighborbincount(self, X):
        projected_X = self.projector.transform(X)
        scores = np.zeros((X.shape[0], self.depth))
        if NJOBS > 0:
            f = lambda chain, proj: chain.neighborbincount(proj)
            with pp.ProcessPool(NJOBS) as pool:
                results = pool.uimap(f, self.chains, [projected_X]*self.nchains)
                pbar = tqdm.tqdm(total=self.nchains, desc='N-Bincount...')
                for s in results:
                    scores += s
                    pbar.update()
        else:
            for i in tqdm.tqdm(range(self.nchains), desc='N-Bincount...'):
                chain = self.chains[i]
                scores += chain.neighborbincount(projected_X)
        scores /= float(self.nchains)
        return scores

    def lociscore(self, X):
        projected_X = self.projector.transform(X)
        scores = np.zeros((X.shape[0], self.depth))
        for i in tqdm.tqdm(range(self.nchains), desc='LOCI...'):
            chain = self.chains[i]
            scores += chain.lociscore(projected_X)
        scores /= float(self.nchains)
        return scores

    def lociscore_adjusted(self, X):
        projected_X = self.projector.transform(X)
        scores = np.zeros((X.shape[0], self.depth))
        if NJOBS > 0:
            f = lambda chain, proj: chain.lociscore_adjusted(proj)
            with pp.ProcessPool(NJOBS) as pool:
                results = pool.uimap(f, self.chains, [projected_X]*self.nchains)
                pbar = tqdm.tqdm(total=self.nchains, desc='N-LOCI...')
                for s in results:
                    scores += s
                    pbar.update()
        else:
            for i in tqdm.tqdm(range(self.nchains), desc='N-LOCI...'):
                chain = self.chains[i]
                scores += chain.lociscore_adjusted(projected_X)
        scores /= float(self.nchains)
        return scores

    def score(self, X, adjusted=False):
        projected_X = self.projector.transform(X)
        scores = np.zeros(X.shape[0])
        if NJOBS > 0:
            f = lambda chain, proj: chain.score(proj, adjusted)
            with pp.ProcessPool(NJOBS) as pool:
                results = pool.uimap(f, self.chains, [projected_X]*self.nchains)
                pbar = tqdm.tqdm(total=self.nchains, desc='Scoring...')
                for s in results:
                    scores += s
                    pbar.update()
        else:
            for i in tqdm.tqdm(range(self.nchains), desc='Scoring...'):
                chain = self.chains[i]
                scores += chain.score(projected_X, adjusted)
        scores /= float(self.nchains)
        return scores
