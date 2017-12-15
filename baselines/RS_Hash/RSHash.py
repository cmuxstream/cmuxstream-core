import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, scale
from scipy.io import loadmat
import pickle

DATA_DIR  = "/nfshome/SHARED/BENCHMARK_HighDim_DATA/Consolidated"

class RSHash(object):
    
    def __init__(self,
                 data,
                 labels,
                 num_components=100,
                 sampling_points=128,
                 num_hash_fns=1,
                 random_state=None,
                 verbose=0):
        self.m = num_components
        self.w = num_hash_fns
        self.s = min(sampling_points,data.shape[0])
        self.X = data
        self.labels = labels
        self.scores = []
        
    def multi_runs(self):
        for i in range(self.m):
            self.scores.append(self.single_run())
    
    def single_run(self):
        hash_functions=[]
        for i in range(self.w):
            hash_functions.append({})
        # Select the locality parameter
        f = np.random.uniform(low=1.0/np.sqrt(self.s), high = 1-1.0/np.sqrt(self.s))
        
        # Generate a d-dimensional random vectors
        alpha = np.zeros(self.X.shape[1],)
        for i in range(self.X.shape[1]):
            alpha[i] = np.random.uniform(low=0, high = f)
        
        # Select integer r (dimensions) to be extracted from dataset
        low = 1+0.5 * (np.log(self.s)/np.log(max(2,1.0/f)))
        high = np.log(self.s)/np.log(max(2,1.0/f))
        r = int(np.random.uniform(low,high))
        if(r>self.X.shape[1]):
            r = self.X.shape[1]
        
        # Select r dimensions from the dataset.
        V = np.random.choice(range(self.X.shape[1]),r,replace=False)
        
        # Randomly sample dataset S of s points.
        selected_indexes = np.random.choice(range(self.X.shape[0]), self.s, replace=False)
        S = self.X[selected_indexes,:]
        
        # Normalize S
        minimum = np.min(S, axis=0)
        maximum = np.max(S, axis=0)
        norm_S =  (S -  minimum)/(maximum -  minimum)
        
        # Shift and Set Y
        Y = -1 * np.ones([S.shape[0], S.shape[1]])
        
        #for i in range(S.shape[0]):
        #    for j in range(S.shape[1]):
        #        if j in V:
        #            Y[i,j] = np.floor((norm_S[i,j] + alpha[j])/f)
        
        for j in range(Y.shape[1]):
            if j in V:
                Y[:,j] = np.floor((norm_S[:,j]+alpha[j])/float(f))
        
        # Apply w different hash functions
        for vec in Y:
            vec = tuple(vec.astype(np.int))
            for i in range(self.w):
                if (vec in hash_functions[i].keys()):
                    hash_functions[i][vec]+=1
                else:
                    hash_functions[i][vec]=1
                    
        # Score the sample
        # Transform each point
        norm_X =  (self.X -  minimum)/(maximum -  minimum)
        score_Y = -1 * np.ones([self.X.shape[0], self.X.shape[1]])
            
        for j in range(score_Y.shape[1]):
            if j in V:
                score_Y[:,j] = np.floor((norm_X[:,j]+alpha[j])/float(f))
        
        score_arr=[]
        for index in range(score_Y.shape[0]):
            vec = score_Y[index]            
            vec = tuple(vec.astype(np.int))
            c = [0]*self.w
            for i in range(self.w):
                if (vec in hash_functions[i].keys()):
                    c[i] = hash_functions[i][vec]
                else:
                    c[i] = 0
            
            if index in selected_indexes:
                score_arr.append(np.log2(min(c)))
            else:
                score_arr.append(np.log2(min(c)+1))
        
        return np.array(score_arr)
    
def read_dataset(filename):
    df = pd.read_csv(filename)
    pt_ids = np.array(df['point.id'])
    labels = np.array(df['ground.truth'])
    labels = [0 if labels[i] == 'nominal' else 1 for i in range(len(labels))]
    labels = np.array(labels)
    X = np.array(df.iloc[:,6:len(df.columns)])
    return X, labels
    
def run_RSHash(X, labels):
    rs_obj = RSHash(X,labels)
    rs_obj.multi_runs()
    anomaly_scores = np.mean(rs_obj.scores,axis=0)
    ap = average_precision_score(labels, anomaly_scores)
    auc = roc_auc_score(labels, anomaly_scores)
    return auc, ap, anomaly_scores

def run_for_consolidated_benchmarks(in_dir, out_file, num_runs=50):
    fw=open(out_file,'w')
    list_files = os.listdir(in_dir)
    for in_file in list_files:
        X, labels = read_dataset(os.path.join(in_dir,in_file))
        auc_arr = []
        ap_arr = []
        for i in range(num_runs):
            if(i%5==0):
                print i
            auc, ap = run_RSHash(X, labels)
            auc_arr.append(auc)
            ap_arr.append(ap)
        fw.write(str(in_file)+","+str(np.mean(auc_arr))+","+str(np.std(auc_arr))+","+str(np.mean(ap_arr))+","+str(np.std(ap_arr))+"\n")
    fw.close()

def run_for_syn_data(num_runs, out_file):
    fw=open(out_file, 'w')
    out_file2=out_file+"_Scores.pkl"
    data = loadmat("../../data/synData.mat")
    X = data['X']
    y = data['y'].ravel()

    X = MinMaxScaler().fit_transform(X)
    
    Xnoisy = np.concatenate((X, np.random.normal(loc=0.5, scale=0.05, size=(X.shape[0], 100))), axis=1)
    X = Xnoisy
    auc_arr = []
    ap_arr = []
    score_arr = []
    for i in range(num_runs):
        if(i%5==0):
            print i
        auc, ap, scores = run_RSHash(X, y)
        auc_arr.append(auc)
        ap_arr.append(ap)
        score_arr.append(scores)
        fw.write(str(auc)+"\t"+str(ap)+"\n")
    fw.close()
    pickle.dump(score_arr, out_file2)
    
#ds_name = "abalone"
#run_for_benchmarks(ds_name)
#in_dir = "/nfshome/SHARED/BENCHMARK_HighDim_DATA/Consolidated"
out_file = "../../Results/RSHash_100.txt"
#run_for_consolidated_benchmarks(in_dir,out_file)
run_for_syn_data(100, out_file)
#in_dir = "/nfshome/SHARED/BENCHMARK_HighDim_DATA/Consolidated"
#out_file = "/nfshome/hlamba/HighDim_OL/Results/RSHash_50.txt"
#run_for_consolidated_benchmarks(in_dir,out_file)
    
    