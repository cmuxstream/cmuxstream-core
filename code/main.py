#!/usr/bin/env python

from constants import *
from HistogramForest import HistogramForest
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

RED = '#e31a1c'
BLUE = '#1f78b4'

if __name__ == "__main__":
    data = loadmat("../data/synData.mat")
    X = data['X']
    y = data['y'].ravel()
    anomalies = y == 1

    Xnoisy = np.concatenate((X, np.random.normal(loc=0.5, scale=0.05,
                                                 size=(X.shape[0], 100))), axis=1)

    plt.figure(0)
    """
    # xstream
    print "Fitting xstream..."
    xtreme_clf = Xtreme(binwidth=10000.0, sketchsize=k, maxdepth=20,
                        ncomponents=1, ntrees=100, deltamax=0.5)
    xtreme_clf.fit(X)
    yscore_xtreme = xtreme_clf.predict(X)

    print "lociplot..."
    xtreme_clf.lociplot(X, y)

    precision, recall, _ = precision_recall_curve(y, yscore_xtreme, pos_label=1)
    average_precision = average_precision_score(y, yscore_xtreme)
    plt.plot(recall, precision, lw=1,
             label='xstream ' + '{:.3f}'.format(average_precision))
    """

    # HistogramForest
    print "Fitting HistForest..."
    ws = np.arange(2.0, 12.0, 2.0)
    ws = [4.0
    scores = np.zeros((Xnoisy.shape[0], len(ws)), dtype=np.float)
    k = 25
    T = 100
    for i, w in enumerate(ws):
        lf = HistogramForest(ntrees=T, w=w, k=k)
        lf.fit(Xnoisy)
        #scores[:,i] = lf.score(X)
        #s = - scores[:,i]
        s = -lf.score(Xnoisy)
        precision, recall, _ = precision_recall_curve(y, s, pos_label=1)
        average_precision = average_precision_score(y, s)
        plt.plot(recall, precision, lw=1, label="HF w=" + str(w) + " AP=" +
                 '{:.3f}'.format(average_precision))
        print "\tk=", k, "T=", T, "w=", w, "AP=", average_precision

    # iforest on full data
    print "Fitting iforest..."
    iforest_clf = IsolationForest(n_estimators=100,
                                  max_samples=1.0,
                                  contamination=0.03,
                                  max_features=1.0,
                                  bootstrap=False,
                                  n_jobs=-1,
                                  random_state=SEED,
                                  verbose=0)
    iforest_clf.fit(X)
    ypred_iforest = (iforest_clf.predict(X) == -1).astype(np.int) # -1 = outlier 
    yscore_iforest = 0.5 - iforest_clf.decision_function(X)
    precision, recall, _ = precision_recall_curve(y, yscore_iforest, pos_label=1)
    average_precision = average_precision_score(y, yscore_iforest)
    plt.plot(recall, precision, lw=1,
             label='IF AP=' + '{:.3f}'.format(average_precision))

    for k in [25, 50, 75]:
        print "Fitting iforest-p..."
        projected_X = SparseRandomProjection(n_components=k,
                                             density=1/3.0,
                                             random_state=SEED).fit_transform(Xnoisy)
        iforest_clf = IsolationForest(n_estimators=100,
                                      max_samples=1.0,
                                      contamination=0.03,
                                      max_features=1.0,
                                      bootstrap=False,
                                      n_jobs=-1,
                                      random_state=SEED,
                                      verbose=0)
        iforest_clf.fit(projected_X)
        yscore_iforest = 0.5 - iforest_clf.decision_function(projected_X)
        precision, recall, _ = precision_recall_curve(y, yscore_iforest, pos_label=1)
        average_precision = average_precision_score(y, yscore_iforest)
        plt.plot(recall, precision, lw=1,
                 label='IF-P k=' + str(k) + ' AP=' + '{:.3f}'.format(average_precision))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.legend()
    plt.savefig("prcurves.png", bbox_inches="tight")
    plt.savefig("prcurves.pdf", bbox_inches="tight")
