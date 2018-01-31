# Authors: Shubhomoy Das
# License: BSD 3 clause

from __future__ import division

import logging
import copy
import os, sys
import numpy as np
import scipy as sp
from warnings import warn

from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix, vstack

from numpy import random

from scipy.sparse import issparse

import numbers
from sklearn.externals import six
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import check_random_state, check_array

from sklearn.ensemble.bagging import BaseBagging
from sklearn.tree._tree import Tree

from multiprocessing import Pool
sys.path.append('../support')
from r_support import *

__all__ = ["HSSplitter", "HSTrees", "HSTree"]

INTEGER_TYPES = (numbers.Integral, np.int)

IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
INFINITY = np.inf
EPSILON = np.finfo('double').eps


class SplitContext(object):
    def __init__(self, min_vals=None, max_vals=None):
        self.min_vals = min_vals
        self.max_vals = max_vals

    def clone(self):
        sd = copy.deepcopy(self)
        return sd

    def __str__(self):
        tmp = cbind(self.min_vals, self.max_vals)
        return "%s" % str(np.transpose(tmp))


class SplitRecord(object):
    def __init__(self, feature=0, threshold=0, pos=0, impurity_right=0, impurity_left=0):
        self.feature = feature
        self.threshold = threshold
        self.pos = pos
        self.impurity_right = impurity_right
        self.impurity_left = impurity_left
        self.left_context = None
        self.right_context = None


class StackRecord(object):
    def __init__(self, start, end, depth, parent, is_left,
                 impurity=0.0, n_constant_features=0, split_context=None):
        self.start = start
        self.end = end
        self.depth = depth
        self.parent = parent
        self.is_left = is_left
        self.impurity = impurity
        self.n_constant_features = n_constant_features
        self.split_context = split_context


class Node(object):
    def __init__(self):
        self.left_child = -1
        self.right_child = -1
        self.feature = -1
        self.threshold = -1
        self.impurity = -1
        self.n_node_samples = -1
        self.weighted_n_node_samples = -1
        ##
        #self.r_node_samples = -1
        #self.l_node_samples = -1

    def __str__(self):
        return "feature: %d, thres: %3.8f, n_node_samples: %3.2f, left: %d, right: %d" % \
                (self.feature, self.threshold, self.n_node_samples, self.left_child, self.right_child)

    def __repr__(self):
        return "feature[%d], thres[%3.8f], n_node_samples[%3.2f], left[%d], right[%d]" % \
                (self.feature, self.threshold, self.n_node_samples, self.left_child, self.right_child)


class ArrTree(object):
    """
    Array-based representation of a binary decision tree.
    
        Attributes
        ----------
        node_count : int
            The number of nodes (internal nodes + leaves) in the tree.
    
        capacity : int
            The current capacity (i.e., size) of the arrays, which is at least as
            great as `node_count`.
    
        max_depth : int
            The maximal depth of the tree.
    
        children_left : array of int, shape [node_count]
            children_left[i] holds the node id of the left child of node i.
            For leaves, children_left[i] == TREE_LEAF. Otherwise,
            children_left[i] > i. This child handles the case where
            X[:, feature[i]] <= threshold[i].
    
        children_right : array of int, shape [node_count]
            children_right[i] holds the node id of the right child of node i.
            For leaves, children_right[i] == TREE_LEAF. Otherwise,
            children_right[i] > i. This child handles the case where
            X[:, feature[i]] > threshold[i].
    
        feature : array of int, shape [node_count]
            feature[i] holds the feature to split on, for the internal node i.
    
        threshold : array of double, shape [node_count]
            threshold[i] holds the threshold for the internal node i.
    
        value : array of double, shape [node_count, n_outputs, max_n_classes]
            Contains the constant prediction value of each node.
    
        impurity : array of double, shape [node_count]
            impurity[i] holds the impurity (i.e., the value of the splitting
            criterion) at node i.
    
        n_node_samples : array of int, shape [node_count]
            n_node_samples[i] holds the number of training samples reaching node i.
    
        weighted_n_node_samples : array of int, shape [node_count]
            weighted_n_node_samples[i] holds the weighted number of training samples
            reaching node i.
            
        r_node_samples : array of int, shape [node_count]
            r_node_samples[i] holds the number of training samples reaching node i in reference.
        l_node_samples : 
            l_node_samples[i] holds the number of training samples reaching node i in latest.
    """
    def __init__(self, n_features, max_depth=0):
        self.n_features = n_features
        self.max_depth = max_depth

        self.node_count = 0
        self.capacity = 0

        self.nodes = None
        self.children_left = None
        self.children_right = None
        self.feature = None
        self.threshold = None
        self.value = None
        self.impurity = None
        self.n_node_samples = None
        #self.r_node_samples = None
        #self.l_node_samples = None
        self.weighted_n_node_samples = None

        self.value_stride = None

        self.clear()

    def clear(self):
        self.nodes = np.zeros(0, dtype=int)
        self.children_left = np.zeros(0, dtype=int)
        self.children_right = np.zeros(0, dtype=int)
        self.feature = np.zeros(0, dtype=int)
        self.threshold = np.zeros(0, dtype=float)
        self.value = np.zeros(0, dtype=float)
        self.impurity = np.zeros(0, dtype=float)
        self.n_node_samples = np.zeros(0, dtype=float)
        self.weighted_n_node_samples = np.zeros(0, dtype=float)
        #self.r_node_samples = np.zeros(0, dtype=float)
        #self.l_node_samples = np.zeros(0, dtype=float)

    def str_node(self, node_id):
        return "feature: %d, thres: %3.8f, n_node_samples: %3.2f, left: %d, right: %d" % \
               (self.feature[node_id], self.threshold[node_id], self.n_node_samples[node_id],
                self.children_left[node_id], self.children_right[node_id])

    def resize(self, capacity=-1):
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.
        """
        # below code is from Cython implementation in sklearn
        self.resize_c(capacity)

    def resize_c(self, capacity=-1):
        """ Guts of resize """

        # below code is from Cython implementation in sklearn
        if capacity == self.capacity and self.nodes is not None:
            return 0

        if capacity == -1:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        self.nodes = np.resize(self.nodes, capacity)
        self.children_left = np.resize(self.children_left, capacity)
        self.children_right = np.resize(self.children_right, capacity)
        self.feature = np.resize(self.feature, capacity)
        self.threshold = np.resize(self.threshold, capacity)
        self.value = np.resize(self.value, capacity)
        self.impurity = np.resize(self.impurity, capacity)
        self.n_node_samples = np.resize(self.n_node_samples, capacity)
        self.weighted_n_node_samples = np.resize(self.weighted_n_node_samples, capacity)
        #self.r_node_samples = np.resize(self.r_node_samples, capacity)
        #self.l_node_samples = np.resize(self.l_node_samples, capacity)

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity

        return 0

    def reset_n_node_samples(self):
        self.n_node_samples[:] = 0
        #self.r_node_samples[:] = 0
        #self.l_node_samples[:] = 0

    def add_node(self, parent, is_left, is_leaf, feature,
                 threshold, impurity, n_node_samples,
                 weighted_n_node_samples):
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        node_id = self.node_count

        # below is from Cython implementation
        if node_id >= self.capacity:
            if self.resize_c() != 0:
                return -1

        self.nodes[node_id] = node_id
        self.impurity[node_id] = impurity
        self.n_node_samples[node_id] = n_node_samples
        #self.r_node_samples[node_id] = n_node_samples
        self.weighted_n_node_samples[node_id] = weighted_n_node_samples

        if parent != TREE_UNDEFINED:
            if is_left:
                self.children_left[parent] = node_id
            else:
                self.children_right[parent] = node_id

        if is_leaf:
            self.children_left[node_id] = TREE_LEAF
            self.children_right[node_id] = TREE_LEAF
            self.feature[node_id] = TREE_UNDEFINED
            self.threshold[node_id] = TREE_UNDEFINED
        else:
            # left_child and right_child will be set later
            self.feature[node_id] = feature
            self.threshold[node_id] = threshold

        self.node_count += 1

        return node_id

    def add_node_samples(self, X):
        if self.node_count < 1:
            # no nodes; likely tree has not been constructed yet
            raise ValueError("Tree not constructed yet")
        for i in np.arange(X.shape[0]):
            node = 0  # start at root
            while node >= 0:
                if node > 0:  # not root
                    self.n_node_samples[node] += 1
                    #self.r_node_samples[node] += 1
                v = X[i, self.feature[node]]
                if self.children_left[node] == -1 and self.children_right[node] == -1:
                    # reached leaf
                    # self.n_node_samples[node] += 1
                    break
                if v <= self.threshold[node]:
                    next_node = self.children_left[node]
                else:
                    next_node = self.children_right[node]
                node = next_node

    def apply(self, X, getleaves=True, getnodeinds=False):
        if self.node_count < 1:
            # no nodes; likely tree has not been constructed yet
            raise ValueError("Tree not constructed yet")
        n = X.shape[0]
        leaves = None
        if getleaves:
            leaves = np.zeros(n, dtype=int)
        x_tmp = None
        if getnodeinds:
            nodeinds = csr_matrix((0, self.node_count), dtype=float)
            x_tmp = lil_matrix((n, self.node_count), dtype=nodeinds.dtype)
        for i in np.arange(n):
            node = 0  # start at root
            while node >= 0:
                if getnodeinds:
                    x_tmp[i, node] = 1
                v = X[i, self.feature[node]]
                if self.children_left[node] == -1 and self.children_right[node] == -1:
                    # reached leaf
                    if getleaves:
                        leaves[i] = node
                    break
                if v <= self.threshold[node]:
                    next_node = self.children_left[node]
                else:
                    next_node = self.children_right[node]
                node = next_node
        if getnodeinds:
            nodeinds = None
            nodeinds = x_tmp.tocsr()
            return leaves, nodeinds
        return leaves

    def __repr__(self):
        s = ''
        pfx = '-'
        stack = list()
        stack.append((0, 0))
        while len(stack) > 0:
            node_id, depth = stack.pop()
            # logger.debug(node_id)
            s = s + "%s%s\n" % (pfx*depth, self.str_node(node_id))
            if self.children_left[node_id] != -1:
                stack.append((self.children_left[node_id], depth + 1))
            if self.children_right[node_id] != -1:
                stack.append((self.children_right[node_id], depth + 1))
        return s

    def __str__(self):
        return self.__repr__()


class HSSplitter(object):
    """
    Attributes:
        split_context: SplitContext
    """
    def __init__(self, random_state=None):
        self.n_samples = 0
        self.weighted_n_samples = None
        self.split_context = None
        self.random_state = random_state

    @staticmethod
    def get_feature_ranges(X, rnd):
        """
        :param X: np.ndarray
        :return: (np.array, np.array) 
        """
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        diff = max_vals - min_vals
        sq = rnd.uniform(0, 1, len(min_vals))
        # logger.debug("sq: %s" % (str(sq)))
        sq_mn = sq - 2 * np.maximum(sq, 1 - sq)
        sq_mx = sq + 2 * np.maximum(sq, 1 - sq)
        mn = min_vals + diff * sq_mn
        mx = min_vals + diff * sq_mx
        return mn, mx

    def init(self, X, y, sample_weight_ptr, X_idx_sorted):
        self.n_samples = X.shape[0]
        self.random_state = check_random_state(self.random_state)
        min_vals, max_vals = HSSplitter.get_feature_ranges(X, self.random_state)
        self.split_context = SplitContext(min_vals, max_vals)
        # logger.debug("root feature ranges:\n%s" % str(self.split_context))

    def node_reset(self, split_context, weighted_n_node_samples=None):
        self.split_context = split_context

    def node_split(self, impurity, split, n_constant_features):
        # select a random feature and split it in half
        feature = self.random_state.randint(0, len(self.split_context.min_vals))
        # logger.debug("splitting %d [%f, %f]" % (feature, self.split_context.min_vals[feature], self.split_context.max_vals[feature]))
        threshold = 0.5 * (self.split_context.min_vals[feature] + self.split_context.max_vals[feature])
        split.feature = feature
        split.threshold = threshold

        split.left_context = self.split_context.clone()
        split.left_context.max_vals[feature] = threshold
        split.right_context = self.split_context.clone()
        split.right_context.min_vals[feature] = threshold


class HSTreeBuilder(object):
    """
    Attributes:
        splitter: HSSplitter
        max_depth: int
    """
    def __init__(self, splitter,
                 max_depth):
        self.splitter = splitter
        self.max_depth = max_depth

    def build(self, tree, X, y, sample_weight=None, X_idx_sorted=None):
        """Build a decision tree from the training set (X, y).
        
        Args:
            tree: ArrTree
            X: numpy.ndarray
            y: numpy.array
            sample_weight: numpy.array
            X_idx_sorted: numpy.array
        """

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree.resize(init_capacity)

        splitter = self.splitter
        max_depth = self.max_depth
        sample_weight_ptr = None

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        n_node_samples = splitter.n_samples
        weighted_n_node_samples = None

        first = 1
        max_depth_seen = -1
        split = SplitRecord()
        stack = list()

        stack.append(StackRecord(0, n_node_samples, 0, TREE_UNDEFINED, 0,
                                 INFINITY, 0, splitter.split_context))

        while len(stack) > 0:
            stack_record = stack.pop()

            start = stack_record.start
            end = stack_record.end
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left
            impurity = stack_record.impurity
            n_constant_features = stack_record.n_constant_features
            split_context = stack_record.split_context

            # logger.debug("feature ranges:\n%s" % str(split_context))

            splitter.node_reset(split_context)

            if first:
                first = 0

            is_leaf = (depth >= max_depth)

            if not is_leaf:
                splitter.node_split(impurity, split, n_constant_features)

            node_id = tree.add_node(parent, is_left, is_leaf, split.feature,
                                    split.threshold, impurity, n_node_samples,
                                    weighted_n_node_samples)
            # logger.debug("Node: %s" % str(tree.nodes[node_id]))

            if not is_leaf:
                # Push right child on stack
                stack.append(StackRecord(split.pos, end, depth + 1, node_id, 0,
                             split.impurity_right, n_constant_features, split.right_context))

                # Push left child on stack
                stack.append(StackRecord(start, split.pos, depth + 1, node_id, 1,
                             split.impurity_left, n_constant_features, split.left_context))

            if False and parent >= 0:
                logger.debug("Parent Node: %s" % str(tree.nodes[parent]))

            if depth > max_depth_seen:
                max_depth_seen = depth

            # tree.resize_c(tree.node_count)
            tree.max_depth = max_depth_seen

        tree.reset_n_node_samples()
        tree.add_node_samples(X)


class HSTree(object):
    def __init__(self,
                 criterion,
                 splitter,
                 max_depth,
                 max_features,
                 random_state):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.n_features_ = None
        self.n_outputs_ = None
        self.classes_ = None
        self.n_classes_ = None

        self.tree_ = None
        self.max_features_ = None

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):

        n_samples, self.n_features_ = X.shape

        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)

        splitter = self.splitter

        self.n_outputs_ = 1
        self.n_classes_ = [1] * self.n_outputs_
        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)
        self.tree_ = ArrTree(self.n_features_)

        builder = HSTreeBuilder(splitter, max_depth)
        builder.build(self.tree_, X, y)

    def apply(self, X):
        return self.tree_.apply(X, getleaves=True, getnodeinds=False)

    def decision_function(self, X):
        """Average anomaly score of X."""
        leaves, nodeinds = self.tree_.apply(X, getleaves=True, getnodeinds=True)
        depths = np.array(np.transpose(nodeinds.sum(axis=1)))
        scores = self.tree_.n_node_samples[leaves] * (2. ** depths)
        
        return scores


def f(i):
    return i * i


def hstree_fit(args):
    max_depth = args[0]
    X = args[1]
    random_state = args[2]
    hst = HSTree(None, HSSplitter(random_state), max_depth, X.shape[1], random_state)
    hst.fit(X, None)
    return hst


def hstree_decision(args):
    X = args[0]
    hst = args[1]
    tree_id = args[2]
    tm = Timer()
    scores = hst.decision_function(X)
    # logger.debug(tm.message("completed HSTree[%d] decision function" % tree_id))
    return scores


class HSTrees(object):
    """Half-Space Trees (HSTrees)

    Return the anomaly score of each sample using the HSTrees algorithm

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    min_vals : list of float, optional (default=None)
        The minimum value for each feature/dimension
        This list must be of the same length as the number of data dimensions
    
    max_vals : list of float, optional (default=None)
        The maximum value for each feature/dimension.
        This list must be of the same length as the number of data dimensions.
    
    max_depth: int
        The maximum depth to which to grow the tree
    
    bootstrap : boolean, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.
    
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.


    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : integer
        The actual number of samples

    References
    ----------
    .. [1] 

    """

    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 max_features=1.,
                 min_vals=None,
                 max_vals=None,
                 max_depth=10,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        self.max_samples=max_samples
        self.max_features=max_features
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = None

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def _fit(self, X, y, max_samples, max_depth, sample_weight=None):
        n_trees = self.n_estimators
        n_pool = self.n_jobs

        p = Pool(n_pool)
        rnd_int = self.random_state.randint(42)
        trees = p.map(hstree_fit, [(max_depth, X, rnd_int + i) for i in range(n_trees)])
        p.close()
        p.join()
        
        return trees

    def fit(self, X, y=None, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : numpy.ndarray
            array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        Returns
        -------
        self : object
            Returns self.
        """
        # ensure_2d=False because there are actually unit test checking we fail
        # for 1d.
        X = check_array(X, accept_sparse=['csc'], ensure_2d=True)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        self.random_state = check_random_state(self.random_state)
        y = self.random_state.uniform(size=X.shape[0])

        # ensure that max_sample is in [1, n_samples]:
        n_samples = X.shape[0]

        self.max_samples_ = n_samples

        self.estimators_ = self._fit(X, y, self.max_samples,
                                     max_depth=self.max_depth,
                                     sample_weight=sample_weight)

        if False:
            for i, estimator in enumerate(self.estimators_):
                logger.debug("Estimator %d:\n%s" % (i, str(estimator.tree_)))
                logger.debug("Node samples:\n%s" % str(estimator.tree_.n_node_samples))

        return self

    def predict(self, X):
        """Predict if a particular sample is an outlier or not."""
        raise NotImplementedError("predict() is not supported for HSTrees")

    def decision_function(self, X):
        """Average anomaly score of X of the base classifiers."""
        n_pool = self.n_jobs
        p = Pool(n_pool)
        scores = np.zeros((1, X.shape[0]))
        tm = Timer()
        if True:
            hst_scores = p.map(hstree_decision, [(X, hst, i) for i, hst in enumerate(self.estimators_)])
        else:
            hst_scores = list()
            for tree_id, hst in enumerate(self.estimators_):
                tm_tree = Timer()
                hst_scores.append(hst.decision_function(X))
                logger.debug(tm_tree.message("completed HSTree[%d] decision function" % tree_id))
        logger.debug(tm.message("completed HSTrees decision_function"))
        for s in hst_scores:
            scores += s
        scores /= len(hst_scores)
        return scores.reshape((scores.shape[1],))

