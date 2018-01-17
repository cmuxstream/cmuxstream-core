#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

indexes = []
aps = []
with open("kdd99_scores.txt.bak", "r") as f:
    for line in f:
        ix, ap = map(float, line.strip().split("\t"))
        indexes.append(ix)
        aps.append(ap)
indexes = np.array(indexes, dtype=np.int)
indexes /= 10000
aps = np.array(aps)

print indexes
print aps

plt.figure()
plt.grid()
plt.xlabel(r"input index (10,000s)")
plt.ylabel("average precision")
plt.ylim([0.0,1.0])
plt.xticks(indexes)
plt.plot(indexes, aps, 'o-', label="XS")
plt.savefig("kddcup99.pdf")
