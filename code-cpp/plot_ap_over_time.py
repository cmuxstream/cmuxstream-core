#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import sys

if __name__ == "__main__":
    ap_files_original = sys.argv[1:]
    ap_files = [f.split("/")[-1] for f in ap_files_original]
    print "AP files:", ap_files

    ds = [int(f.split("_")[-3][1:]) for f in ap_files]
    Is = [20.0, 25.0, 20.0, 25.0]

    plt.figure()
    plt.grid()
    plt.xlabel("tuple index")
    plt.ylabel("average precision")
    plt.ylim((0.0,1.0))

    for idx, file in enumerate(ap_files_original):
        nrows = []
        aps = []
        d = ds[idx]
        I = Is[idx]

        with open(file, "r") as f:
            for line in f:
                fields = line.strip().split(" ")
                r = int(fields[0])
                ap = float(fields[1])
                nrows.append(r)
                aps.append(ap)

        plt.plot(nrows, aps, 'o-', fillstyle='none',
                 label="XS-C (d=" + str(d) + ", N=" + str(I) + ")")

    plt.legend()
    plt.savefig("ap_over_time.pdf", bbox_inches="tight")
