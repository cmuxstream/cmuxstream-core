#!/usr/bin/env python

from constants import *
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from HSTree import HSTree
from HSTrees import HSTrees
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Histogram:

    def __init__(self, deltamax=20.0, levels=15):
        self.deltamax = deltamax
        self.levels = levels
        self.fs = []
        self.cmsketches = []

    def fit(self, X):
        ndim = X.shape[1]
        nrows = X.shape[0]
        levelcount = {f: -1 for f in range(ndim)}
        binids = np.zeros(X.shape, dtype=np.float)
        for level in range(self.levels):
            f = np.random.randint(0, ndim) # random feature
            self.fs.append(f)
            levelcount[f] += 1

            if levelcount[f] == 0:
                binids[:,f] = X[:,f]/self.deltamax
            else:
                binids[:,f] = binids[:,f] * 2.0

            cmsketch = {}
            for row in binids:
                l = tuple(np.floor(row).astype(np.int))
                if not l in cmsketch:
                    cmsketch[l] = 0
                cmsketch[l] += 1

            self.cmsketches.append(cmsketch)
            #print "Level counts:", level
            #for k, v in cmsketch.iteritems():
            #    print v

    def score(self, X):
        scores = np.zeros((X.shape[0], self.levels))
        
        ndim = X.shape[1]
        levelcount = {f: -1 for f in range(ndim)}
        binids = np.zeros(X.shape, dtype=np.float)
        for level in range(self.levels):
            f = self.fs[level]
            levelcount[f] += 1

            if levelcount[f] == 0:
                binids[:,f] = X[:,f]/self.deltamax
            else:
                binids[:,f] = binids[:,f] * 2.0

            cmsketch = self.cmsketches[level]
            for i, row in enumerate(binids):
                l = tuple(np.floor(row).astype(np.int))
                if not l in cmsketch:
                    scores[i,level] = 0
                else:
                    scores[i,level] = cmsketch[l]

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

        # create feature map
        ndim = self.sketchsize
        fmap = {}
        cmap = {}
        for i in range(ndim):
            fmap[i] = set([])
            k = np.random.randint(0, ndim/2 + 1)
            #k = self.ncomponents #FIX
            for j in range(k):
                c_j = np.random.randint(0, self.ncomponents)
                #c_j = j #FIX
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

            t = HSTrees(n_estimators=self.ntrees, max_samples=1.0, n_jobs=1,
                        random_state=SEED, max_depth=10)
            t.fit(component_X)
            trees.append(t)

            """
            print "Component:", j
            tk = []
            for k in range(50):
                t = Histogram(levels=10)
                t.fit(component_X)
                tk.append(t)
            trees.append(tk)
            """

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
        #yscore = self.trees.decision_function(projected_X)

        yscore = np.zeros(X.shape[0])
        for j, t_j in enumerate(self.trees):
            component_X = projected_X[:, self.cmap[j]]
            yscore_j = t_j.decision_function(component_X)

            """
            yscore_j = np.zeros(X.shape[0])
            for hist in t_j:
                npoints = hist.score(component_X) # score for each level
                hist_score = 1.0/np.mean(npoints, axis=1) # aggreg. across levels
                yscore_j += hist_score
            yscore_j /= float(len(t_j))
            """
            yscore += yscore_j
        yscore /= self.ncomponents
        yscore = -yscore

        return ypred, yscore

    def lociplot(self, X, y):
        projected_X = self.projector.fit_transform(X)
        yscore = np.zeros(X.shape[0])
        for j, t_j in enumerate(self.trees):
            #score from each component
            component_X = projected_X[:, self.cmap[j]]
            #yscore_j = t_j.decision_function(component_X)

            yscore_j = np.zeros((X.shape[0], 12))
            for tree in t_j.estimators_:
                tree_score = tree.decision_function_loci(X)
                yscore_j += tree_score
            yscore_j /= float(len(t_j.estimators_))
            yscore_j = -yscore_j

            plt.figure(5)

            anomalies = y == 1
            mean_anomaly_scores = np.median(yscore_j[anomalies,:], axis=0)
            pct25_anomaly_scores = np.percentile(yscore_j[anomalies,:], q=25.0,
                                                 axis=0)
            pct75_anomaly_scores = np.percentile(yscore_j[anomalies,:], q=75.0,
                                                 axis=0)
            
            mean_benign_scores = np.median(yscore_j[~anomalies,:], axis=0)
            pct25_benign_scores = np.percentile(yscore_j[~anomalies,:], q=25.0,
                                                axis=0)
            pct75_benign_scores = np.percentile(yscore_j[~anomalies,:], q=75.0,
                                                axis=0)
 
            plt.plot(range(12), mean_anomaly_scores.T, '-', color='#e41a1c',
                     alpha=1.0)
            plt.plot(range(12), pct25_anomaly_scores.T, '--', color='#e41a1c',
                     alpha=0.5)
            plt.plot(range(12), pct75_anomaly_scores.T, '--', color='#e41a1c',
                     alpha=0.5)
            plt.plot(range(12), mean_benign_scores.T, '-', color='#377eb8',
                     alpha=1.0)
            plt.plot(range(12), pct25_benign_scores.T, '--', color='#377eb8',
                     alpha=0.5)
            plt.plot(range(12), pct75_benign_scores.T, '--', color='#377eb8',
                     alpha=0.5)

            plt.xticks(range(12))
            plt.xlabel(r'Level $d$')
            plt.ylabel(r'Anomaly score $-N.d \times 2^d$')
            plt.grid()
            plt.savefig("lociplot-mean.png", bbox_inches="tight")

            plt.clf()
            plt.plot(range(12), yscore_j[~anomalies,:].T, '-o', alpha=0.1,
                     color='#377eb8')
            plt.plot(range(12), yscore_j[anomalies,:].T, '-o', alpha=0.2,
                     color='#e41a1c')
            plt.xticks(range(12))
            plt.xlabel(r'Level $d$')
            plt.ylabel(r'Anomaly score $-N.d \times 2^d$')
            plt.grid()
            plt.savefig("lociplot.png", bbox_inches="tight")
