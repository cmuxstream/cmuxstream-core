#!/usr/bin/env python

import numpy as np
from xtreme import Histogram
import matplotlib.pyplot as plt

X = np.array([[0.1, 0.8], # top left
              [0.2, 0.7], # top left
              [0.7, 0.2], # bot right
              [0.8, 0.1], # bot right
              [0.8, 0.8]])

plt.plot(X[:,0], X[:,1], 'o')
plt.grid()
plt.savefig("histogram_test_scatter.png", bbox_inches="tight")

h = Histogram(deltamax=0.5, levels=3)
h.fit(X, verbose=True)

print "scores"
scores = h.score(X)
print scores
