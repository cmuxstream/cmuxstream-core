#!/usr/bin/env python

from constants import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Chains import Chains
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import sys
from tqdm import tqdm

if __name__ == "__main__":
    data = loadmat("../data/synDataNoisy.mat")
    X = data["X"]
    y = data["y"].ravel()

    xs = []
    ys = []
    with open("../code-cpp/synDataScoresovertime.txt", "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            nrows = int(fields[0])

            if np.sum(y[:nrows]) == 0:
                continue

            scores = map(float, fields[1].split(" ")) 
            scores = -np.array(scores)
            ap = average_precision_score(y[:nrows], scores) 

            print nrows, ap
