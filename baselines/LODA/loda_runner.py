import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, scale
from loda import *
from loda_support import *
from ensemble_support import *
from scipy.io import loadmat
import pickle
import time

def read_dataset2(filename):
    data = np.loadtxt(filename, delimiter=',')
    n,m = data.shape
    X = data[:,0:m-1]
    y = data[:,m-1]
    print n,m, X.shape, y.shape
    
    return X,y

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

def run_LODA(X, labels):
    #lodares = loda(X, sparsity=0.3, mink=1, maxk=103)
    lodares = loda(X, maxk=110)
    #model = generate_model_from_loda_result(lodares, X, labels)
    #anoms, lbls, _, _, _, detector_scores, detector_wts = (
    #        model.anoms, model.lbls,
    #        model.w, model.hists, model.hpdfs, model.nlls, model.proj_wts
    #    )
    auc, ap = compute_statistics(lodares.nll, labels)
    print "AUC="+str(auc)+ " & AP="+str(ap)
    return auc, ap, lodares.nll
    
def run_for_consolidated_benchmarks(in_dir, out_file, num_runs):
    out_file2=out_file+"_Scores.pkl"
    fw=open(out_file,'w')
    list_files = os.listdir(in_dir)
    for in_file in list_files:
        print "Running for:"+str(in_file)
        X, labels = read_dataset(os.path.join(in_dir,in_file))
        auc_arr = []
        ap_arr = []
        score_arr = []
        for i in range(num_runs):
            if(i%5==0):
                print "\t\t"+str(i)
            auc, ap, scores = run_LODA(X, labels)
            auc_arr.append(auc)
            ap_arr.append(ap)
            score_arr.append(scores)
            fw.write(str(i)+"\t"+str(auc)+"\t"+str(ap)+"\n")
    fw.write(str(in_file)+","+str(np.mean(auc_arr))+","+str(np.std(auc_arr))+","+str(np.mean(ap_arr))+","+str(np.std(ap_arr))+"\n")
    fw.close()
    pickle.dump(score_arr, open(out_file2,"w"))

def run_for_syn_data(num_runs, out_file):
    fw=open(out_file, 'w')
    out_file2=out_file+"_Scores.pkl"
    
    data = loadmat("../../data/synDataNoisy.mat")
    X = data['X']
    y = data['y'].ravel()

    #X = MinMaxScaler().fit_transform(X)
    
    #Xnoisy = np.concatenate((X, np.random.normal(loc=0.5, scale=0.05, size=(X.shape[0], 100))), axis=1)
    #X = Xnoisy
    auc_arr = []
    ap_arr = []
    score_arr = []
    for i in range(num_runs):
        if(i%5==0):
            print i
        auc, ap, scores = run_LODA(X, y)
        auc_arr.append(auc)
        ap_arr.append(ap)
        score_arr.append(scores)
        fw.write(str(auc)+"\t"+str(ap)+"\n")
    fw.close()
    pickle.dump(score_arr, open(out_file2,"w"))
    
#ds_name = "abalone"
#run_for_benchmarks(ds_name)
#in_dir = "/nfshome/SHARED/BENCHMARK_HighDim_DATA/Consolidated"

#in_dir = "/home/SHARED/BENCHMARK_HighDim_DATA/Consolidated_Irrel"
#out_file = "../../../Results_Irrel/NEW_LODA_50.txt"
#run_for_consolidated_benchmarks(in_dir,out_file,50)
#run_for_syn_data(100, out_file)    
#out_file = "/nfshome/hlamba/HighDim_OL/Results/LODA_50.txt"
#run_for_consolidated_benchmarks(in_dir,out_file)

def run_for_dataset(in_file, out_file, num_runs):
    fw=open(out_file,'w')
    out_file2=out_file+"_Scores.pkl"
    print "Doing for:"+str(in_file)
    X, labels = read_dataset2(os.path.join(in_dir,in_file))
    auc_arr = []
    ap_arr = []
    score_arr = []
    for i in range(num_runs):
        if(i%5==0):
            print "\t\t"+str(i)
        auc, ap, scores = run_LODA(X, labels)
        auc_arr.append(auc)
        ap_arr.append(ap)
        score_arr.append(scores)
        fw.write(str(i)+"\t"+str(auc)+"\t"+str(ap)+"\n")
    fw.write(str(np.mean(auc_arr))+","+str(np.std(auc_arr))+","+str(np.mean(ap_arr))+","+str(np.std(ap_arr))+"\n")
    fw.close()
    pickle.dump(score_arr, open(out_file2,"w"))
#in_dir = "/home/SHARED/BENCHMARK_HighDim_DATA/Consolidated_Irrel"
#out_dir = "../../Results_Irrel/NEW_LODA"
in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets"
out_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/temp"

in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/LowDim"
out_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Results/LowDim_Option1/LODA"

print "Running LODA"
file_name = sys.argv[1]
num_runs = int(sys.argv[2])
in_file = os.path.join(in_dir,file_name)
out_file = os.path.join(out_dir, file_name)
start_time = time.time()
run_for_dataset(in_file, out_file, num_runs)
print "Time Taken="+str(time.time() - start_time)+ " for:"+str(file_name)
