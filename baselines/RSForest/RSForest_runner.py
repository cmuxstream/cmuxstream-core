import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from random_split_trees import RSForest

DATA_DIR = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets"

def read_dataset(filename):
    df = pd.read_csv(filename)
    pt_ids = np.array(df['point.id'])
    labels = np.array(df['ground.truth'])
    labels = [0 if labels[i] == 'nominal' else 1 for i in range(len(labels))]
    labels = np.array(labels)
    X = np.array(df.iloc[:,6:len(df.columns)])
    return X, labels
    
def compute_statistics(scores, labels):
    avg_precision = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)
    return auc, avg_precision
    
def run_RSForest(X, labels):
    clf = RSForest(n_estimators=100)
    print "Classifier Initialized"
    clf.fit(X)
    print "Classifier Fit."
    scores = clf.decision_function(X)
    auc, ap = compute_statistics(scores, labels)
    return auc, ap
        
def get_index(in_file):
    in_file = in_file[in_file.rfind("_")+1:len(in_file)]
    index = in_file.replace(".csv","")
    return index
    
def run_for_benchmarks(ds_name):
    data_path = os.path.join(DATA_DIR, ds_name)
    list_files = os.listdir(data_path)
    auc_arr = ap_arr = indexes = []
    for in_file in list_files:
        index = get_index(in_file)
        X, labels = read_dataset(os.path.join(data_path,in_file))
        print "Dataset Read"
        auc, ap = run_RSForest(X, labels)
        print index, auc, ap
        indexes.append(index)
        auc_arr.append(auc)
        ap_arr.append(ap)
    
    print index, auc_arr, ap_arr
    
ds_name = "abalone"
run_for_benchmarks(ds_name)