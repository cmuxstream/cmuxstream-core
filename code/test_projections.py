#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist
from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import precision_recall_curve, average_precision_score

if __name__ == "__main__":
    data = loadmat("../data/synData.mat")
    X = data["X"]
    y = data["y"].ravel()

    Xnoisy = np.concatenate((X, np.random.normal(loc=0.5, scale=0.05,
                                                 size=(X.shape[0], 100))), axis=1)

    plt.figure(0)
    orig_dists = pdist(Xnoisy, metric='euclidean').ravel()
    print max(orig_dists)
    pd = []
    for i, k in enumerate(range(50,90,10)):
        print "k:", k
        projector = SparseRandomProjection(n_components=k,
                                           density=1/3.0,
                                           random_state=SEED)
        projected_X = projector.fit_transform(Xnoisy)

        projected_dists = pdist(projected_X, metric='euclidean')
        pd.append(projected_dists)

	iforest_clf = IsolationForest(n_estimators=100,
                                      max_samples=1.0,
                                      contamination=0.03,
                                      max_features=1.0,
                                      bootstrap=False,
                                      n_jobs=-1,
                                      random_state=SEED,
                                      verbose=0)
        iforest_clf.fit(projected_X)
        yscore = 0.5 - iforest_clf.decision_function(projected_X)
        precision, recall, _ = precision_recall_curve(y, yscore, pos_label=1)
        average_precision = average_precision_score(y, yscore)

        plt.plot(recall, precision, lw=1,
                 label='k=' + str(k) + ' ap= ' + '{:.3f}'.format(average_precision))
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()

    plt.legend()
    plt.savefig("pr_curve_vs_k.png", bbox_inches="tight")

    for i, k in enumerate(range(50,90,10)):
        print "k:", k
        projected_dists = pd[i].ravel()
        nonzero = orig_dists != 0

        plt.clf()
        plt.hexbin(orig_dists[nonzero], projected_dists[nonzero],
                   gridsize=100, cmap=plt.cm.Blues, bins='log')
        plt.xlabel("Original distances (Euclidean)")
        plt.ylabel("Projected distances")
        cb = plt.colorbar()
        cb.set_label("Pair count")
        plt.savefig("dist_histogram_k" + str(k) + ".png", bbox_inches="tight")
