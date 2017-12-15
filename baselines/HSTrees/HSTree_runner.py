import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from HSTrees import HSTrees

DATA_DIR = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/"

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
    
def run_HSTrees(X, labels):
    clf = HSTrees(n_estimators=100, max_samples="auto")
    print "Classifier Initialized"
    clf.fit(X)
    print "Classifier Fit."
    scores = clf.decision_function(X)
    auc, ap = compute_statistics(scores, labels)
    del clf
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
        auc, ap = run_HSTrees(X, labels)
        print index, auc, ap
        indexes.append(index)
        auc_arr.append(auc)
        ap_arr.append(ap)
    
    print index, auc_arr, ap_arr
    
def run_for_consolidated_benchmarks(in_dir, out_file, num_runs=10):
    fw=open(out_file,'w')
    list_files = os.listdir(in_dir)
    for in_file in list_files:
        X, labels = read_dataset(os.path.join(in_dir,in_file))
        auc_arr = []
        ap_arr = []
        for i in range(num_runs):
            if(i%5==0):
                print i
            auc, ap = run_HSTrees(X, labels)
            auc_arr.append(auc)
            ap_arr.append(ap)
        fw.write(str(in_file)+","+str(np.mean(auc_arr))+","+str(np.std(auc_arr))+","+str(np.mean(ap_arr))+","+str(np.std(ap_arr))+"\n")
    fw.close()

def run_for_syn_data(num_runs, out_file):
    fw=open(out_file, 'w')
    data = loadmat("../data/synData.mat")
    X = data['X']
    y = data['y'].ravel()

    X = MinMaxScaler().fit_transform(X)
    
    Xnoisy = np.concatenate((X, np.random.normal(loc=0.5, scale=0.05, size=(X.shape[0], 100))), axis=1)
    X = Xnoisy
    auc_arr = []
    ap_arr = []
    for i in range(num_runs):
        if(i%5==0):
            print i
        auc, ap = run_HSTrees(X, y)
        auc_arr.append(auc)
        ap_arr.append(ap)
    fw.write(str(in_file)+","+str(np.mean(auc_arr))+","+str(np.std(auc_arr))+","+str(np.mean(ap_arr))+","+str(np.std(ap_arr))+"\n")
    fw.close()
    
#ds_name = "abalone"
#run_for_benchmarks(ds_name)
#in_dir = "/nfshome/SHARED/BENCHMARK_HighDim_DATA/Consolidated"
out_file = "../../Results/HSTree_100.txt"
#run_for_consolidated_benchmarks(in_dir,out_file)
run_for_syn_data(100, out_file)