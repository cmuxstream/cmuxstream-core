#!/usr/bin/env python

from constants import *
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from HSTree import HSTree
from HSTrees import HSTrees

class Histogram:

    def __init__(self, deltamax=0.5, levels=5):
        self.deltamax = deltamax
        self.levels = levels
        self.fs = [None]
        self.cmsketches = []

    def fit(self, X):
        # level 0
        binned_X = X/self.deltamax

        cmsketch = {}
        for row in binned_X:
            l = tuple(np.floor(row).astype(np.int))
            if not l in cmsketch:
                cmsketch[l] = 0
            cmsketch[l] += 1
        self.cmsketches.append(cmsketch)

        # other levels
        ndim = X.shape[1]
        for level in range(1, self.levels):
            cmsketch = {}
            f = np.random.randint(0, ndim) # random feature
            self.fs.append(f)

            divider = np.ones(ndim, dtype=np.float)
            divider[f] = 2.0 
            binned_X = binned_X/divider

            for row in binned_X:
                l = tuple(np.floor(row).astype(np.int))
                if not l in cmsketch:
                    cmsketch[l] = 1
                cmsketch[l] += 1

            self.cmsketches.append(cmsketch)

    def score(self, X):
        scores = np.zeros((X.shape[0], self.levels))

        # level 0
        binned_X = X/self.deltamax

        for i, row in enumerate(binned_X):
            l = tuple(np.floor(row).astype(np.int))
            if not l in self.cmsketches[0]:
                scores[i,0] = 0
            else:
                scores[i,0] = self.cmsketches[0][l]

        # other levels
        ndim = X.shape[1]
        for level in range(1, self.levels):
            f = self.fs[level]
            divider = np.ones(ndim, dtype=np.float)
            divider[f] = 2.0 
            binned_X = binned_X/divider

            for row in binned_X:
                l = tuple(np.floor(row).astype(np.int))
                if not l in self.cmsketches[level]:
                    scores[i,level] = 0
                else:
                    scores[i,level] = self.cmsketches[level][l]

        return scores

class Xtreme:

    def __init__(self, binwidth=1.0, sketchsize=15, ntrees=100, maxdepth=10,
                 ncomponents=10):
        # n_components >= 4 log(n_samples) / (eps^2 / 2 - eps^3 / 3)
        # See Theorem 2.1:
        #   http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.3654
        #
        # For eps = 0.1, n_components >= n_features
        self.projector = SparseRandomProjection(n_components=sketchsize,
                                                density=1/3.0,
                                                random_state=SEED)
        self.binwidth = binwidth
        self.ntrees = ntrees
        self.maxdepth = maxdepth
        self.ncomponents = ncomponents
        self.sketchsize = sketchsize
        self.trees = None

    def fit(self, X):
        assert self.projector.n_components < X.shape[1]

        """
        for i in range(20):
            # sample 50 random columns with replacement
            cols = np.unique(np.random.randint(X.shape[1], size=50))
            self.cols.append(cols)

            Xsample = X[:,cols]
            projected_X = self.projector.fit_transform(Xsample)
        
            trees = []
            for t in range(50):
                #print "fitting tree", t
                tree = HSTree(maxdepth=10)
                tree = tree.fit(projected_X)
                trees.append(tree)
            self.sampled_trees.append(trees)
        """
        # create feature map
        ndim = self.sketchsize
        fmap = {}
        cmap = {}
        for i in range(ndim):
            fmap[i] = set([])
            k = np.random.randint(0, ndim/2 + 1)
            for j in range(k):
                c_j = np.random.randint(0, self.ncomponents)
                fmap[i].add(c_j)
                if c_j not in cmap:
                    cmap[c_j] = set([])
                cmap[c_j].add(i)
            fmap[i] = np.array(fmap[i])
        self.fmap = fmap
        self.cmap = cmap

        projected_X = self.projector.fit_transform(X)

        trees = []
        for j in range(self.ncomponents):
            cmap[j] = np.array(list(cmap[j]), dtype=np.int) 
            component_X = projected_X[:, cmap[j]]
            
            """
            t = HSTrees(n_estimators=self.ntrees, max_samples=1.0, n_jobs=16,
                        random_state=SEED)
            t.fit(component_X)
            trees.append(t)
            """

            t = Histogram()
            t.fit(component_X)
            trees.append(t)
        self.trees = trees

        """
        trees = HSTrees(n_estimators=100, max_samples=1.0, n_jobs=16,
                        random_state=SEED)
        trees.fit(projected_X)
        self.trees = trees
        """
        
        """
        trees = []
        for t in range(self.ntrees):
            #print "fitting tree", t
            tree = HSTree(maxdepth=self.maxdepth)
            tree = tree.fit(projected_X)
            trees.append(tree)
        self.trees = trees
        """

    def predict(self, X, threshold=0.5):
        """ 1 if outlier, 0 otherwise """
        ypred = np.ones(X.shape[0], dtype=np.int)
        yscore = np.zeros(X.shape[0], dtype=np.float)
        samplescore = np.zeros((X.shape[0], 20), dtype=np.float)

        #assert self.cmsketch is not None
        #assert self.trees is not None
        #assert self.sampled_trees is not None
        
        """
        for i in range(20):
            # sample 50 random columns with replacement
            cols = self.cols[i] 
            Xsample = X[:,cols]
            trees = self.sampled_trees[i]
            projected_X = self.projector.fit_transform(Xsample)
        
            for j in range(X.shape[0]):
                scores = np.zeros(len(trees))
                for k, t in enumerate(trees):
                    scores[k] = t.score(projected_X[j,:])
                samplescore[j,i] = np.mean(scores)

        yscore = np.max(samplescore, axis=1)
        print yscore.shape
        """

        """
        projected_X = self.projector.fit_transform(X)
        for i in range(X.shape[0]):
            scores = np.zeros(len(self.trees))
            for j, t in enumerate(self.trees):
                scores[j] = t.score(projected_X[i,:])
            yscore[i] = np.mean(scores)
        """

        projected_X = self.projector.fit_transform(X)
        #yscore = self.trees.decision_function(projected_X)

        yscore = 0.0
        for j, t_j in enumerate(self.trees):
            component_X = projected_X[:, self.cmap[j]]
            #yscore_j = t_j.decision_function(component_X)
            yscore_j = t_j.score(component_X)
            yscore_j = -np.mean(yscore_j, axis=1)
            yscore += yscore_j
        yscore /= self.ncomponents

        """
        binned_X = np.floor(projected_X/self.binwidth).astype(np.int)
        for i, row in enumerate(binned_X):
            l = tuple(row)
            bincount = 0
            if l in self.cmsketch:
                bincount = self.cmsketch[l]
            yscore[i] = float(bincount)
        """

        return ypred, yscore
