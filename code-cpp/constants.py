#!/usr/bin/env python

SEED = 42
NJOBS = 100

# synthetic data point classes
CLASSES = [range(0,1000), # sparse benign cluster
           range(1000,3000), # dense benign cluster
           range(3000,3050), # clustered anomalies
           range(3050, 3075), # sparse anomalies
           range(3076,3082), # local anomalies
           range(3075,3076)] # single anomaly
