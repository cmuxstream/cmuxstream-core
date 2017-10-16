import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from HSTrees.HSTrees import HSTrees
from RSForest.random_split_trees import RSForest
from LODA.loda import *
from LODA.ensemble_support import *

sys.path.append("./support")

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

def run_TreeBased(X, labels, clf_type):
    if(clf_type == "HS-Tree"):
        clf = HSTrees(n_estimators=100, max_samples="auto")
    elif(clf_type == "RS-Forest"):
        clf = RSForest(n_estimators=100)
    elif(clf_type == "I-Forest"):
        clf = IsolationForest(n_estimators = 1000, max_samples = 10)
    else:
        raise NotImplementedError
    
    clf.fit(X)
    scores = clf.decision_function(X)
    auc, ap = compute_statistics(scores, labels)
    return auc, ap

def run_LODA(X, labels):
    mink = 100
    maxk = 200
    exclude = None
    original_dims = False
    print "running LODA...."
    algo_result = loda(X, mink=max(int(ncol(X)/2), mink), maxk=maxk,
                               keep=None, exclude=None,
                               original_dims=original_dims)
    
    print algo_result.nll.shape
    model = generate_model_from_loda_result(algo_result, X, labels)
    anoms, lbls, _, _, _, detector_scores, detector_wts = (
            model.anoms, model.lbls,
            model.w, model.hists, model.hpdfs, model.nlls, model.proj_wts)
    print model.anom_score
    auc, ap = compute_statistics(-model.anom_score, labels)
    return auc, ap
    
def get_index(in_file):
    in_file = in_file[in_file.rfind("_")+1:len(in_file)]
    index = in_file.replace(".csv","")
    return index
    
def run_for_benchmarks(ds_name,clf_type):
    data_path = os.path.join(DATA_DIR, ds_name)
    list_files = os.listdir(data_path)
    auc_arr = ap_arr = indexes = []
    for in_file in list_files:
        index = get_index(in_file)
        X, labels = read_dataset(os.path.join(data_path,in_file))
        print "Size of dataset="+str(X.shape)
        #auc, ap = run_TreeBased(X, labels, clf_type)
        auc, ap = run_LODA(X, labels)
        print index, auc, ap
        print HI
        indexes.append(index)
        auc_arr.append(auc)
        ap_arr.append(ap)
    
    print index, auc_arr, ap_arr

def run_for_single_dataset(ds_path, clf_type):
    X, labels = read_dataset(os.path.join(data_path,in_file))
    print "Size of dataset="+str(X.shape)
    #auc, ap = run_TreeBased(X, labels, clf_type)
    auc, ap = run_LODA(X, labels)
    
ds_name = "abalone"
clf_type = "I-Forest"   #HS-Tree, RS-Forest, I-Forest
run_for_benchmarks(ds_name, clf_type)