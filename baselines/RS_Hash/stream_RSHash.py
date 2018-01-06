import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, scale
from scipy.io import loadmat
import pickle
import time

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

class RSHash_Stream(object):
    
    def __init__(self,
                 data,
                 labels,
                 decay_rate = 0.015,
                 num_components=100,
                 sampling_points=1000,
                 window_size=256,
                 num_hash_fns=1,
                 random_state=None,
                 verbose=0):
        self.m = num_components
        self.w = num_hash_fns
        self.s = min(sampling_points,data.shape[0])
        self.X = data
        self.labels = labels
        self.decay_rate = decay_rate
        self.scores = []
        
        self.pre_algo_stats()
        
    def pre_algo_stats(self):
        self.minimum = self.X.min(axis=0)
        self.maximum = self.X.max(axis=0)
        self.s_effective = max(self.s, 1/ (1 - (np.power(2,self.decay_rate))))
        
        self.f_arr=[]
        self.V_arr=[]
        self.alphas=[]
        self.hash_functions = []
        
        for i in range(self.w):
            self.hash_functions.append({})
                
        for i in range(self.m):
            f = np.random.uniform(low = 1.0/np.sqrt(self.s_effective), high = 1 -1.0/np.sqrt(self.s_effective)) 
            self.f_arr.append(f)
            
            low = 1+0.5 * (np.log(self.s_effective)/np.log(max(2,1.0/f)))
            high = np.log(self.s_effective)/np.log(max(2,1.0/f))
            r = int(np.random.uniform(low,high))
            
            if(r>self.X.shape[1]):
                r = self.X.shape[1]
            # Subset of dimensions Vr is sampled upfront.
            V = np.random.choice(range(self.X.shape[1]),r,replace=False)
            selected_V = V[np.where(self.minimum[V]!=self.maximum[V])]
            self.V_arr.append(selected_V)
            
            alpha = np.zeros(selected_V.shape[0],)
            for j in range(len(selected_V)):
                alpha[j] = np.random.uniform(low=0, high = f)
                
            self.alphas.append(alpha)
            
    def fit(self):
        init_scores=[]
        for i in range(self.m):
            selected_indexes = np.random.choice(range(self.X.shape[0]), self.s, replace=False)
            S = self.X[selected_indexes,:]
            
            norm_S =  (S -  self.minimum)/(self.maximum -  self.minimum)
            norm_S[np.abs(norm_S) == np.inf] = 0
            
            Y = -1 * np.ones([S.shape[0], S.shape[1]])
            for j in range(Y.shape[1]):
                index_j = np.where(self.V_arr[i] == j)[0]
                if j in self.V_arr[i]:
                    Y[:,j] = np.floor((norm_S[:,j]+self.alphas[i][index_j])/float(self.f_arr[i]))
            
            for index in range(len(Y)):
                curr_tstamp = selected_indexes[index]
                vec = Y[index]
                vec = tuple(vec.astype(np.int))
                #Appending the component number.
                vec = vec + (i,)
                for h in range(self.w):
                    if (vec in self.hash_functions[h].keys()):
                        curr_value = self.hash_functions[h][vec]
                        magnitude = curr_value[0]
                        node_timestamp = curr_value[1]
                        if curr_tstamp > node_timestamp:
                            node_timestamp = curr_tstamp 
                        self.hash_functions[h][vec] = (magnitude+1, node_timestamp)
                    else:
                        self.hash_functions[h][vec]=(1, curr_tstamp)
            
            
            norm_X =  (self.X -  self.minimum)/(self.maximum -  self.minimum)
            norm_X[np.abs(norm_X) == np.inf] =0
            score_Y = -1 * np.ones([self.X.shape[0], self.X.shape[1]])
            
            for j in range(score_Y.shape[1]):
                if j in self.V_arr[i]:
                    index_j = np.where(self.V_arr[i] == j)[0]
                    score_Y[:,j] = np.floor((norm_X[:,j]+self.alphas[i][index_j])/float(self.f_arr[i]))
        
            print "Scoring Now for:"+str(i)
            score_arr=[]
            for index in range(score_Y.shape[0]):
                vec = score_Y[index]
                vec = tuple(vec.astype(np.int))
                vec = vec + (i,)
                c = [0]*self.w
                for h in range(self.w):
                    if (vec in self.hash_functions[h].keys()):
                        c[h] = self.hash_functions[h][vec][0]
                    else:
                        c[h] = 0
            
                if index in selected_indexes:
                    score_arr.append(np.log2(min(c)))
                else:
                    score_arr.append(np.log2(min(c)+1))
        
            init_scores.append(score_arr)
        
        avg_scores = np.mean(init_scores, axis=0)
        return avg_scores
    
    def stream_score(self, X, curr_tstamp):
        norm_X =  (X -  self.minimum)/(self.maximum -  self.minimum)
        norm_X[np.abs(norm_X) == np.inf] = 0
        stream_scores=[]
        for i in range(self.m):
            score_Y = -1 * np.ones([norm_X.shape[0],])
            for j in range(score_Y.shape[0]):
                index_j = np.where(self.V_arr[i] == j)[0]
                if j in self.V_arr[i]:
                    score_Y[j] = np.floor((norm_X[j]+self.alphas[i][index_j])/float(self.f_arr[i]))
            score_arr=[]
            
            score_arr=[]
            vec = score_Y
            vec = tuple(vec.astype(np.int))
            vec = vec + (i,)
                
            c = [0]*self.w
            for h in range(self.w):
                if (vec in self.hash_functions[h].keys()):
                    curr_value = self.hash_functions[h][vec]
                    magnitude = curr_value[0]
                    node_timestamp = curr_value[1]
                    wt_factor = np.power(2, -self.decay_rate * (curr_tstamp - node_timestamp))
                    c[h] = wt_factor*magnitude
                    self.hash_functions[h][vec] = (c[h]+1, curr_tstamp)
                else:
                    c[h] = 0
                    self.hash_functions[h][vec]=(1, curr_tstamp)
                    
            score_arr.append(np.log2(min(c)+1))
            stream_scores.append(score_arr)
            
        print len(stream_scores)
        avg_scores = np.mean(stream_scores, axis=0)
        return avg_scores
    
def run_RSHashStream(X, labels, init_size):
    X_init = X[0:init_size]
    labels_init = labels[0:init_size]
    rs_obj = RSHash_Stream(X_init,labels_init)
    init_scores = rs_obj.fit()
    auc,ap = compute_statistics(-init_scores, labels_init)
    overall_scores = list(init_scores)
    
    for index in range(init_size, X.shape[0]):
        score = rs_obj.stream_score(X[index], index)
        overall_scores.append(score)        
    
    overall_scores = np.array(overall_scores)
    auc,ap = compute_statistics(-overall_scores,labels)
    return auc,ap, overall_scores

def run_for_dataset(in_file, out_file, num_runs):
    out_file2=out_file+"_Scores.pkl"
    print "Doing for:"+str(in_file)
    X, labels = read_dataset2(os.path.join(in_dir,in_file))
    auc_arr = []
    ap_arr = []
    score_arr = []
    for i in range(num_runs):
        print "\t\t"+str(i)
        auc, ap, scores = run_RSHashStream(X, labels, 256)
        auc_arr.append(auc)
        ap_arr.append(ap)
        score_arr.append(scores)
        np.savetxt(out_file+"_Scores_"+str(i)+".txt",scores)

#in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Streaming_HighDim"
#out_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Results/Streaming_HighDim"
in_dir = "/home/SHARED/BENCHMARK_HighDim_DATA/Streaming_HighDim"
out_dir = "/home/SHARED/BENCHMARK_HighDim_DATA/Results/6_1_18/Streaming_HighDim"

print "Running RSHash"
file_name = sys.argv[1]
#file_name = "madelon_overall.txt_10.0_0.1_15.0_NOISY_Random_0"
#num_runs = int(sys.argv[2])
num_runs = 1
in_file = os.path.join(in_dir,file_name)
out_file = os.path.join(out_dir, file_name)
start_time = time.time()
run_for_dataset(in_file, out_file, num_runs)
print "Time Taken="+str(time.time() - start_time)+ " for:"+str(file_name)
        