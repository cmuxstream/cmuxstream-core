#!/usr/bin/env python

from constants import *
import numpy as np
from sklearn.random_projection import SparseRandomProjection

class LSHForest:

    def __init__(self, ntrees=10, depth=20):
        self.l = ntrees # number of trees
        self.d = depth # projection size
        self.trees = []

    def fit(self, X):
        for i in range(self.l):
            tree = LSHTree(depth=self.d)
            tree.fit(X)
            self.trees.append(tree)

    def score(self, X):
        scores = np.zeros((X.shape[0], self.d), dtype=np.float)
        for tree in self.trees:
            scores += tree.score(X)
        scores /= float(self.l)
        return scores

class LSHTree:

    def __init__(self, depth):
        num_nodes = 2 ** (depth+1) - 1
        self.depth = depth
        self._tree = np.zeros(num_nodes, dtype=np.int)
        self.projector = SparseRandomProjection(n_components=depth,
                                                density=1/3.0,
                                                random_state=SEED)

    def fit(self, X):
        projected_X = self.projector.fit_transform(X)
        for x in projected_X:
            self._insert(x)

    def score(self, X):
        projected_X = self.projector.transform(X)
        scores = np.zeros((X.shape[0], self.depth), dtype=np.float)
        for i, x in enumerate(projected_X):
            scores[i,:] = self._score(x)
        return scores

    def _left(self, idx):
        return 2 * idx + 1
    
    def _right(self, idx):
        return 2 * idx + 2

    def _parent(self, idx):
        return (idx - 1) / 2

    def _insert(self, x):
        current_idx = 0
        for bit in x:
            if bit > 0: # right
                current_idx = self._right(current_idx)
            else: # left
                current_idx = self._left(current_idx)
            self._tree[current_idx] += 1
        return self._tree[current_idx]

    def _score(self, x):
        scores = np.zeros(self.depth, dtype=np.float)
        current_idx = 0
        for i, bit in enumerate(x):
            if bit > 0: # right
                current_idx = self._right(current_idx)
            else: # left
                current_idx = self._left(current_idx)
            scores[i] = self._tree[current_idx]
        return scores
