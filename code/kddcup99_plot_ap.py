#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

indexes = []
aps = []
with open("kdd99_scores.txt", "r") as f:
    for line in f:
        ix, ap = map(float, line.strip().split("\t"))
        indexes.append(ix)
        aps.append(ap)
indexes = np.array(indexes, dtype=np.int)
aps = np.array(aps)

plt.figure()
plt.grid()
plt.xlabel(r"Tuple Index")
plt.ylabel("Average Precision")
plt.ylim([0.0,1.0])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),
                     useMathText=True)
plt.plot(indexes, aps, 'o-', fillstyle='none', label="XS")
plt.legend()
plt.savefig("kddcup99.pdf")
