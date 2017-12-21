#!/usr/bin/env python

from scipy.io import loadmat, savemat
import numpy as np

data = loadmat("../data/synData.mat")
X = data["X"]
y = data["y"].ravel()
Xnoisy = np.concatenate((X, np.random.normal(loc=0.5, scale=0.05,
                                             size=(X.shape[0], 100))), axis=1)

savemat("synDataNoisy.mat", {"X": Xnoisy, "y": y})
