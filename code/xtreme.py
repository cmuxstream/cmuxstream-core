#!/usr/bin/env python

from constants import *
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from HSTree import HSTree
from HSTrees import HSTrees

class Xtreme:
    projected_X = None
    projector = None
    trees = None
    sampled_trees = []
    cmsketch = {}
    cols = []

    def __init__(self, binwidth=1.0, sketchsize=15, ntrees=100, maxdepth=10):
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

        projected_X = self.projector.fit_transform(X)
        """
        binned_X = np.floor(projected_X/self.binwidth).astype(np.int)
        for row in binned_X:
            l = tuple(row)
            if not l in self.cmsketch:
                self.cmsketch[l] = 1
            self.cmsketch[l] += 1
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

        trees = HSTrees(n_estimators=100, max_samples='auto')
        trees.fit(projected_X)
        self.trees = trees

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
        yscore = self.trees.decision_function(projected_X)

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
