#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.random_projection import SparseRandomProjection
from scipy.io import loadmat
from scipy.stats import kurtosis
from scipy.spatial.distance import pdist, squareform
import sys
from xtreme import Xtreme
from HSTree import HSTree

RED = '#e31a1c'
BLUE = '#1f78b4'

def plot_distance_histograms(X, tag):
    dists = squareform(pdist(X, 'euclidean'))

    # inlier distance histogram
    inlier_distances = dists[y==0,:]
    np.fill_diagonal(inlier_distances, -1)
    inlier_distances = inlier_distances[inlier_distances>0].ravel()
    plt.figure(0)
    plt.hist(inlier_distances, bins=50,
             align='mid')
    #plt.xticks(np.arange(0.0,1.5,0.1))
    plt.grid()
    plt.savefig("inlier_hist_" + tag + ".png", bbox_inches="tight")
    plt.clf()

    outlier_distances = dists[y==1,:]
    np.fill_diagonal(outlier_distances, -1)
    outlier_distances = outlier_distances[outlier_distances>0].ravel()
    plt.figure(0)
    plt.hist(outlier_distances, bins=50,
             align='mid', color=RED)
    #plt.xticks(np.arange(0.0,1.5,0.1))
    plt.grid()
    plt.savefig("outlier_hist_" + tag + ".png", bbox_inches="tight")
    plt.clf()

if __name__ == "__main__":
    if sys.argv[1] == "synth":
        #data = loadmat("../data/speech.mat")
        data = loadmat("../data/synData.mat")
        #data = loadmat("../data/synData-withNoise.mat")
        X = data['X']
        y = data['y'].ravel()

        X = MinMaxScaler().fit_transform(X)
        
        # add noise
        Xnoisy = np.concatenate((X, np.random.normal(loc=0.0, scale=0.05,
                                                     size=(X.shape[0], 100))), axis=1)
        X = Xnoisy
    elif sys.argv[1] == "http":
        data = np.genfromtxt("../data/kddcup.data", delimiter=",")
        X = data[:,:-1]
        y = data[:,-1]
        y[y=="normal"] = 0
        y[y!="normal"] = 1

    k = 50
    projector = SparseRandomProjection(n_components=k,
                                         density=1/3.0,
                                         random_state=SEED)
    projected_X = projector.fit_transform(X)

    plot_distance_histograms(X, tag="nonoise")
    plot_distance_histograms(projected_X, tag="projected" + str(k))

    kurt = kurtosis(X, fisher=False)
    plt.figure(0)
    plt.hist(kurt, bins=50, align='mid')
    plt.grid()
    plt.savefig("kurt_hist.png", bbox_inches="tight")
    plt.clf()
    kurt = kurtosis(projected_X, fisher=False)
    plt.figure(0)
    plt.hist(kurt, bins=50, align='mid')
    plt.grid()
    plt.savefig("kurt_hist_projected.png", bbox_inches="tight")
    plt.clf()

    xtreme_clf = Xtreme(binwidth=10000.0, sketchsize=k, maxdepth=10)
    xtreme_clf.fit(X)
    ypred_xtreme, yscore_xtreme = xtreme_clf.predict(X) 
    yscore_xtreme = -1.0 * yscore_xtreme

    plt.figure(0)
    precision, recall, _ = precision_recall_curve(y, yscore_xtreme, pos_label=1)
    average_precision = average_precision_score(y, yscore_xtreme)
    plt.plot(recall, precision, lw=1,
             label='xstream ' + '{:.3f}'.format(average_precision))

    # iforest on full data
    iforest_clf = IsolationForest(n_estimators=100,
                                  max_samples='auto',
                                  contamination=0.03,
                                  max_features=1.0,
                                  bootstrap=False,
                                  n_jobs=-1,
                                  random_state=SEED,
                                  verbose=0)
    iforest_clf.fit(X)
    ypred_iforest = (iforest_clf.predict(X) == -1).astype(np.int) # -1 = outlier 
    yscore_iforest = -1.0 * iforest_clf.decision_function(X)

    plt.figure(0)
    precision, recall, _ = precision_recall_curve(y, yscore_iforest, pos_label=1)
    average_precision = average_precision_score(y, yscore_iforest)

    plt.plot(recall, precision, lw=1,
             label='iforest ' + '{:.3f}'.format(average_precision))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()

    if X.shape[1] <= 3:
        plt.legend()
        plt.savefig("pr_xstream_iforest.png", bbox_inches="tight")
        sys.exit()

    iforest_clf = IsolationForest(n_estimators=100,
                                  max_samples='auto',
                                  contamination=0.03,
                                  max_features=1.0,
                                  bootstrap=False,
                                  n_jobs=-1,
                                  random_state=SEED,
                                  verbose=0)

    iforest_clf.fit(projected_X)
    ypred_iforest = (iforest_clf.predict(projected_X) == -1).astype(np.int) # -1 = outlier 
    yscore_iforest = -1.0 * iforest_clf.decision_function(projected_X)

    precision, recall, _ = precision_recall_curve(y, yscore_iforest, pos_label=1)
    average_precision = average_precision_score(y, yscore_iforest)

    plt.plot(recall, precision, lw=1,
             label='iforest-p ' + '{:.3f}'.format(average_precision))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("pr_xstream_iforest.png", bbox_inches="tight")
