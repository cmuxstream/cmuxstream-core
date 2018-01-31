import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

def compute_statistics(scores, labels):
    avg_precision = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)
    return auc, avg_precision

def read_dataset2(filename):
    data = np.loadtxt(filename, delimiter=',')
    n,m = data.shape
    X = data[:,0:m-1]
    y = data[:,m-1]
    print n,m, X.shape, y.shape
    
    return X,y

def plot_auc_over_time(init_size,X,y,scores):
    sc = scores[0:init_size]
    labels = y[0:init_size]
    
    auc_arr = []
    ap_arr = []
    x_arr = [init_size]
    
    auc, ap = compute_statistics(-sc, labels)
    auc_arr.append(auc)
    ap_arr.append(ap)
    
    for i in range(init_size,len(scores)):
        sc = scores[0:init_size+i]
        labels = y[0:init_size+i]
        auc, ap = compute_statistics(-sc, labels)
        
        auc_arr.append(auc)
        ap_arr.append(ap)
        x_arr.append(i)
    
    return auc_arr,ap_arr,x_arr
    
def read_scores(score_file):
    scores = np.loadtxt(score_file)
    return scores

def plot(auc_arr,ap_arr,x_arr,plots_file):
    plt.figure(figsize=(8,6))
    plt.plot(x_arr,ap_arr, linewidth=3.0)
    plt.xlabel('#Points',fontsize=18)
    plt.ylabel('AP', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(plots_file+"_AP.pdf")
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.plot(x_arr,auc_arr, linewidth=3.0)
    plt.xlabel('#Points',fontsize=18)
    plt.ylabel('AUC', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(plots_file+"_AUC.pdf")
    plt.close()

if __name__ == '__main__':
    in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Streaming_HighDim"
    scores_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Results/Streaming_HighDim"
    plots_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Results/Plots_6-1/Streaming_HighDim"
    filename = "madelon_overall.txt_10.0_0.1_15.0_NOISY_Random_0"
    X, y = read_dataset2(os.path.join(in_dir, filename))
    scores = read_scores(os.path.join(scores_dir, filename+"_Scores_0.txt"))
    auc_arr,ap_arr,x_arr = plot_auc_over_time(256,X,y,scores)
    plot(auc_arr,ap_arr,x_arr,os.path.join(plots_dir,filename))
      
    
    
    