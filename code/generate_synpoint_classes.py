#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.io import loadmat

data = loadmat("../data/synDataNoisy.mat")
X = data["X"]
y = data["y"].ravel()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for idx, c in enumerate(CLASSES):
    label = str(idx)
    ax.scatter(X[c,0], X[c,1], X[c,2], 'o', label=label)

plt.legend(ncol=6, loc='upper center', fontsize=8)
plt.grid()
plt.savefig("syndata.pdf", bbox_inches="tight")
