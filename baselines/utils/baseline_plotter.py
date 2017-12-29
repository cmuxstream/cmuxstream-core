from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def read_dataset(in_file):
    data = np.loadtxt(in_file, delimiter=',')
    X = data[:,0:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    return X, y


def get_scores(inp_file):
    data = pickle.load(open(inp_file,"r"))
    sum_arr=np.zeros(data[0].shape)
    for x in data:
        sum_arr = sum_arr + np.array(x)
    
    return sum_arr
    
def get_stats(inp_file):
    f=open(inp_file,"r")
    lines=f.readlines()
    stats=lines[len(lines)-1]
    split=stats.split(",")
    mean_auc=float(split[0])
    std_auc=float(split[1])
    mean_ap=float(split[2])
    std_ap=float(split[3])
    
    return np.array([mean_auc, std_auc, mean_ap, std_ap])

def average_scores_file_highdim(in_dir, ds_file, noise_level, out_file):
    '''
        Plots the PR Curve for HighDim datasets
    '''
    plt.figure(figsize=(10,8))
    algo_arr={}
    algo_names=['iForest','HSTrees','LODA','RSHash']
    X,y = read_dataset(ds_file)
    for algo_name in algo_names:
        inp_file = os.path.join(in_dir,algo_name,ds_name+"_overall.txt_5.0_0.1_"+str(noise_level)+"_NOISY_SCORES.pkl")
        if(algo_name=="LODA"):
            pr1,re1,_ = precision_recall_curve(y,get_scores(inp_file))
        else:
            pr1,re1,_ = precision_recall_curve(y,-get_scores(inp_file))
        plt.plot(re1, pr1, linewidth=3.0)
        
    plt.legend(algo_names,fontsize=20)
    plt.title(ds_name,fontsize=24)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision",fontsize=20)
    plt.savefig(out_file)

def average_scores_file_lowdim(in_dir, ds_file, ds_name, dim, noise_level, out_file):
    plt.figure(figsize=(10,8))
    algo_arr={}
    algo_names=['iForest','HSTrees','LODA','RSHash']
    dims=['100.0', '1000.0', '2000.0', '5000.0']

    X,y = read_dataset(ds_file)
    for algo_name in algo_names:
        inp_file = os.path.join(in_dir,algo_name,ds_name+"_overall.txt_"+str(dim)+"_"+str(noise_level)+"_NOISY_SCORES.pkl")
        if(algo_name=="LODA"):
            pr1,re1,_ = precision_recall_curve(y,get_scores(inp_file))
        else:
            pr1,re1,_ = precision_recall_curve(y,-get_scores(inp_file))
        plt.plot(re1, pr1, linewidth=3.0)
            
    plt.legend(algo_names,fontsize=20)
    plt.title(ds_name,fontsize=24)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision",fontsize=20)
    plt.savefig(out_file)
    
def plot_baselines_highdim(in_dir, ds_name, method, out_file):
    plt.figure(figsize=(12,8))
    algo_arr={}
    algo_names=['iForest','HSTrees','LODA','RSHash']
    dims=['10.0', '20.0', '50.0']
    for algo_name in algo_names:
        res_arr = []
        if algo_name in algo_arr.keys():
            res_arr = algo_arr[algo_name]
#        orig_file = os.path.join(in_dir, algo_name, ds_name+"_overall.txt")
#        res_arr.append(get_stats(orig_file))
        for dim in dims:
            inp_file = os.path.join(in_dir,algo_name,ds_name+"_overall.txt_5.0_0.1_"+str(dim)+"_NOISY")
            res_arr.append(get_stats(inp_file))
            
        algo_arr[algo_name] = res_arr

    x_arr=np.arange(0,len(algo_arr[algo_name])*5,5)
#    xticks=['Original']
    xticks=[]
    xticks = xticks+dims

        
    index=0
    for algo_names in algo_arr:
        a = np.array(algo_arr[algo_names])
        if(method=='AUC'):
            plt.bar(x_arr-index, a[:,0], yerr = a[:,1])
        else:
            plt.bar(x_arr-index, a[:,2], yerr = a[:,3])
        index+=1
        
    plt.legend(list(algo_arr.keys()),fontsize=20)
    plt.xticks(x_arr-2, xticks,fontsize=18)
    plt.title(ds_name+"_"+str(method),fontsize=24)
    plt.savefig(out_file)
    
def plot_baselines_lowdim(in_dir, ds_name, noise_level,method,out_file):
    plt.figure(figsize=(12,8))
    algo_arr={}
    algo_names=['iForest','HSTrees','LODA','RSHash']
    dims=['100.0', '1000.0', '2000.0', '5000.0']
    for algo_name in algo_names:
        res_arr = []
        if algo_name in algo_arr.keys():
            res_arr = algo_arr[algo_name]
        orig_file = os.path.join(in_dir, algo_name, ds_name+"_overall.txt")
        res_arr.append(get_stats(orig_file))
        for dim in dims:
            inp_file = os.path.join(in_dir,algo_name,ds_name+"_overall.txt_"+str(dim)+"_"+str(noise_level)+"_NOISY")
            res_arr.append(get_stats(inp_file))
            
        algo_arr[algo_name] = res_arr

    x_arr=np.arange(0,len(algo_arr[algo_name])*5,5)
    xticks=['Original']
    xticks = xticks+dims

        
    index=0
    for algo_names in algo_arr:
        a = np.array(algo_arr[algo_names])
        if(method=='AUC'):
            plt.bar(x_arr-index, a[:,0], yerr = a[:,1])
        else:
            plt.bar(x_arr-index, a[:,2], yerr = a[:,3])
        index+=1
        
    plt.legend(list(algo_arr.keys()),fontsize=18)
    plt.xticks(x_arr-2, xticks,fontsize=18)
    plt.title(ds_name+"_"+str(method),fontsize=24)
    plt.savefig(out_file)

if __name__ == '__main__':
    in_dir="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Results/HighDim"
    out_dir="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Results/Plots_28-12/HighDim"
    ds_name="pima-indians"
    noise_level="0.1"
    method="AP"
    plot_baselines_lowdim(in_dir, ds_name, noise_level,method,out_file)

    ds_name="letter-recognition"
    out_file=os.path.join(out_dir,ds_name+"_"+str(noise_level)+"_"+str(method)+".pdf")
    plot_baselines_highdim(in_dir, ds_name, method, out_file)
    
    in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Results/LowDim"
    ds_name="magic-telescope"
    noise_level="0.1"
    dim="100.0"
    ds_file="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/LowDim/"+str(ds_name)+"_overall.txt"+"_"+str(dim)+"_"+str(noise_level)+"_NOISY"
    out_file="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Results/Plots_28-12/LowDim/PRCurve_"+str(ds_name)+"_"+str(dim)+".pdf"
    average_scores_file_lowdim(in_dir, ds_file, ds_name, dim, noise_level, out_file)
    
    in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Results/HighDim"
    ds_name="gisette"
    noise_level="10.0"
    ds_file="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/HighDim/"+str(ds_name)+"_overall.txt_5.0_0.1_"+str(noise_level)+"_NOISY"
    out_file="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Results/Plots_28-12/HighDim/PRCurve_"+str(ds_name)+"_"+str(noise_level)+".pdf"
    average_scores_file_highdim(in_dir, ds_file, noise_level, out_file)

