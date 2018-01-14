#!/usr/bin/env python

from constants import *
import itertools
from sklearn.random_projection import SparseRandomProjection
from StreamhashProjection import StreamhashProjection
import pathos.pools as pp
import numpy as np
import time
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

    def lociscore(self, X):
        scores = self.bincount(X)

        multiplier = np.array([2.0 ** d for d in range(1, self.depth+1)])
        scores *= multiplier

        indices = np.arange(scores.shape[0])

        lociscores = np.zeros(scores.shape, dtype=np.float)
        means = np.mean(scores, axis=0)
        lociscores = scores - means

        return lociscores

    def lociscore_density(self, X):
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
            fs = self.fs[:depth+1]
            for i, prebin in enumerate(prebins):
                bin_i = np.floor(prebin).astype(np.int)
                l = tuple(bin_i)
                if not l in cmsketch:
                    scores[i,depth] = 0.0
                else:
                    scores[i,depth] = cmsketch[l]# * 2.0**depth

                neighbor_bincounts = []
                # 1-step
                for f in fs:
                    for delta in [1,-1]:
                        nbr = bin_i[:]
                        nbr[f] += delta
                        nbr = tuple(nbr)
                        if nbr in cmsketch:
                            neighbor_bincounts.append(cmsketch[nbr])
                        else:
                            neighbor_bincounts.append(0)

                if len(fs) == 1:
                    scores[i,depth] -= np.mean(neighbor_bincounts)
                    scores[i,depth] *= 2**depth
                    continue

                """
                # 2-step
                for f1, f2 in itertools.combinations(fs, 2):
                    for delta1, delta2 in itertools.combinations([1, -1, 0], 2):
                        if delta1 == 0 and delta2 == 0:
                            continue
                        nbr = bin_i[:]
                        nbr[f1] += delta1
                        nbr[f2] += delta2
                        nbr = tuple(nbr)
                        if nbr in cmsketch:
                            neighbor_bincounts.append(cmsketch[nbr])
                        else:
                            neighbor_bincounts.append(0)
                """
                prebin_f = np.array([prebin[f] for f in fs])
                left_dist = prebin - np.floor(prebin)
                prob_f = 1.0 - np.minimum(left_dist, 1.0 - left_dist)
                prob_right = left_dist
                for w in range(min(3**depth-1, 50)):
                    dims = np.random.binomial(1, p=prob_f)
                    shifts = 2 * np.random.binomial(1, p=prob_right) - 1
                    perturbation = shifts * dims
                    for j in range(len(fs)):
                        nbr = bin_i[:]
                        nbr[fs[j]] += perturbation[j]
                        nbr = tuple(nbr)
                        if nbr in cmsketch:
                            neighbor_bincounts.append(cmsketch[nbr])
                        else:
                            neighbor_bincounts.append(0)
                scores[i,depth] -= np.mean(neighbor_bincounts)
                scores[i,depth] *= 2**depth

        return scores

    def score(self, X, density=False):
        if density:
            lociscores = self.lociscore_density(X)
        else:
            lociscores = self.lociscore(X)
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

    def lociscore(self, X):
        projected_X = self.projector.transform(X)
        scores = np.zeros((X.shape[0], self.depth))
        for i, chain in enumerate(self.chains):
            print "LOCI chain", i, "..."
            scores += chain.lociscore(projected_X)
        scores /= float(self.nchains)
        return scores

    def lociscore_density(self, X):
        projected_X = self.projector.transform(X)
        scores = np.zeros((X.shape[0], self.depth))
        """
        for i, chain in enumerate(self.chains):
            print "LOCI-d chain", i, "..."
            scores += chain.lociscore_density(projected_X)
        """
        #"""
        def f(chain, proj):
            return chain.lociscore_density(proj)
        pool = pp.ProcessPool(NJOBS)
        results = pool.uimap(f, self.chains, [projected_X]*len(self.chains))
        pbar = tqdm.tqdm(total=len(self.chains), desc='LOCI-d')
        for s in results:
            scores += s
            pbar.update()
        #"""
        scores /= float(self.nchains)
        return scores

    def score(self, X, density=False):
        projected_X = self.projector.transform(X)
        scores = np.zeros(X.shape[0])

        f = lambda chain, proj: chain.score(proj, density)
        with pp.ProcessPool(NJOBS) as pool:
            results = pool.uimap(f, self.chains, [projected_X]*len(self.chains))
            pbar = tqdm.tqdm(total=len(self.chains), desc='Scoring...')
            for s in results:
                scores += s
                pbar.update()
        scores /= float(self.nchains)
        return scores
