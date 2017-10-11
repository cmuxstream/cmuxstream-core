#!/usr/bin/env python

from constants import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import scale
from scipy.io import loadmat
from xtreme import Xtreme

RED = '#e31a1c'
BLUE = '#1f78b4'

if __name__ == "__main__":
    data = loadmat("../data/speech.mat")
    X = data['X']
    y = data['y'].ravel()

    X = scale(X)

    xtreme_clf = Xtreme(binwidth=100000000.0)
    xtreme_clf.fit(X)
    ypred_xtreme, yscore_xtreme = xtreme_clf.predict(X) 

    outlier_idx = y == 1
    outlier_scores = yscore_xtreme[outlier_idx]
    inlier_scores = yscore_xtreme[~outlier_idx]

    ax = plt.subplots()
    counts, bins, patches = plt.hist(outlier_scores, bins=np.arange(0,20),
                                     color=RED)
    plt.grid()
    plt.xticks(np.arange(0,20,1))
    plt.savefig('outlier_scores.png', bbox_inches='tight')
    ax = plt.subplots()
    counts, bins, patches = plt.hist(inlier_scores, bins=np.arange(0,20), 
                                     color=BLUE)
    plt.grid()
    plt.xticks(np.arange(0,20,1))
    plt.savefig('inlier_scores.png', bbox_inches='tight')
    
    print 'XTREME' 
    print '\tPrecision:', precision_score(y, ypred_xtreme)
    print '\tRecall:', recall_score(y, ypred_xtreme)

    iforest_clf = IsolationForest(n_estimators=100,
                                  max_samples='auto',
                                  contamination=0.1,
                                  max_features=1.0,
                                  bootstrap=False,
                                  n_jobs=-1,
                                  random_state=SEED,
                                  verbose=0)
    iforest_clf.fit(X)
    ypred_iforest = (iforest_clf.predict(X) == -1).astype(np.int) # -1 = outlier 

    print 'IFOREST' 
    print '\tPrecision:', precision_score(y, ypred_iforest)
    print '\tRecall:', recall_score(y, ypred_iforest)
