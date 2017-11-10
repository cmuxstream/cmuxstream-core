#!/usr/bin/env python

import numpy as np
from scipy.stats import kurtosis

class HSForest:

    def __init__(self, depth=20, ntrees=50):
        self.depth = depth
        self.ntrees = ntrees
        self.trees = []
        for i in range(ntrees):
            self.trees.append(HSTree())

    def fit(self, X):
        for tree in self.trees:
            tree.fit(X)

    def score(self, X):
        pass

class HSNode:

    def __init__(self, data, workmin, workmax, windows, depth, maxdepth,
                 parent, recurse=True):
        # data
        #self.data = data
        self.n = data.shape[0]
        self.r = 0
        self.l = 0

        # bounding box
        self.workmin = workmin
        self.workmax = workmax
        self.split = (0,0.0) # dimension, location

        # tree
        self.depth = depth
        self.maxdepth = depth
        self.parent = parent
        self.left = None
        self.right = None

        if not recurse or depth == maxdepth:
            #print depth, 'terminal', data.shape[0], workmin, workmax
            return

        ndim = data.shape[1]
        
        # random split
        q = np.random.randint(0, ndim)
        # kurtosis split
        """
        if data.shape[0] > 0:
            k = kurtosis(data, axis=0, fisher=False)
            print k
            if np.sum(k) == 0.0:
                k = [1.0/ndim] * ndim
            else:
                k = k/float(np.sum(k))
            q = np.random.choice(range(ndim), p=k)
        """
        p = (workmax[q] + workmin[q])/2.0
        self.split = (q, p)

        data_l = data[data[:,q]<p]
        data_r = data[data[:,q]>=p]

        #print depth, data.shape[0], workmin, workmax, self.split

        workmax_l = workmax.copy()
        workmax_l[q] = p
        self.left = HSNode(data=data_l, workmin=workmin, workmax=workmax_l,
                           windows=None, depth=depth+1,
                           maxdepth=maxdepth, parent=self, recurse=True)
        
        workmin_l = workmin.copy()
        workmin_l[q] = p
        self.right = HSNode(data=data_r, workmin=workmin_l, workmax=workmax,
                            windows=None, depth=depth+1,
                            maxdepth=maxdepth, parent=self, recurse=True)

    def score(self, x):
        if self.left is None and self.right is None:
            #print 'terminal:', self.data.shape[0], self.depth 
            #return self.data.shape[0] * 2.0**self.depth
            return self.n * 2.0**self.depth

        q, p = self.split
        #print 'Split at:', self.depth, q, p
        if x[q] <= p:
            #print '\tLeft'
            return self.left.score(x)
        else:
            #print '\tRight'
            return self.right.score(x)

class HSTree:

    def __init__(self, maxdepth):
        self.root = None
        self.maxdepth = maxdepth

    def fit(self, X):
        # compute workspace for this tree
        ndim = X.shape[1]
        s = np.random.random([ndim])
        workmin = s - 2.0 * np.maximum(s, 1.0 - s)
        workmax = s + 2.0 * np.maximum(s, 1.0 - s)

        self.root = HSNode(data=X, workmin=workmin, workmax=workmax,
                           windows=None, depth=0, maxdepth=self.maxdepth,
                           parent=None, recurse=True)

        return self

    def score(self, x):
        return self.root.score(x)
