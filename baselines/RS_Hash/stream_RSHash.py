import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, scale
from scipy.io import loadmat
import pickle
import time
from tqdm import tqdm

def compute_statistics(scores, labels):
    avg_precision = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)
    return auc, avg_precision

class Stream_RS_Hash(object):
    
    def __init__(self, data, labels,
                 sampling_points = 1000,
                 decay = 0.015,
                 num_components = 100, 
                 num_hash_fns = 1, 
                 random_state = None):
        
        self.m = num_components
        self.w = num_hash_fns
        self.s = min(sampling_points, data.shape[0])
        self.X = data
        self.n = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.labels = labels
        self.decay = decay
        self.scores = []
        self.num_hash = num_hash_fns
        self.cmsketches = []
        self.effS = max(1000, 1.0/(1 - np.power(2, -self.decay)))
        print "setting s to:"+str(self.s)
        print "setting decay to:"+str(self.decay)
        print "Effective S="+str(self.effS)
        self._preprocess()
        #self.process_first_batch()
        #self.init_scores = self.score_first_batch()
        
    def _preprocess(self):
        for i in range(self.num_hash):
            self.cmsketches.append({})
        print "Number of Sketches="+str(self.cmsketches)
        print "PreProcessing..."
        self._getMinMax()
        print "Min="+str(self.minimum.shape)+" Max="+str(self.maximum.shape)
        self._sample_f()
        print "Sampled quantity f ->"+str(self.f)
        self._sample_dims()
        print "Sample Dimensions:"+str(self.V)
        self._sample_shifts()
        print "Sampling Shifts:"+str(len(self.alpha))
        
    def _getMinMax(self):
        self.pp_data = self.X[:self.s,:]
        self.minimum = np.min(self.pp_data, axis=0)
        self.maximum = np.max(self.pp_data, axis=0)
        print "Min Shape="+str(self.minimum.shape)+" and max shape="+str(self.maximum.shape)        
        
    def _sample_dims(self):
        max_term = np.max((2*np.ones(self.f.size),list(1.0/self.f)),axis=0)
        common_term = np.log(self.effS)/np.log(max_term)
        low_value = 1 + 0.5 * common_term
        high_value = common_term
        self.r = np.empty([self.m,],dtype=int)
        self.V=[]
        for i in range(self.m):
            self.r[i] = min(np.random.randint(low = low_value[i], high = high_value[i]), self.dim)
            sel_V = np.random.choice(range(self.pp_data.shape[1]),size = self.r[i], replace = False)
            self.V.append(sel_V[np.where(self.minimum[sel_V]!=self.maximum[sel_V])])
        
    def _normalize(self):
        self.X_normalized = (self.pp_data - self.minimum)/(self.maximum - self.minimum)
        self.X_normalized[np.abs(self.X_normalized) == np.inf] = 0
        
    def _sample_shifts(self):
        self.alpha = []
        for r in range(self.m):
            self.alpha.append(np.random.uniform(low = 0, high=self.f[r], size = len(self.V[r])))
            
    def _sample_f(self):
        self.f = np.random.uniform(low = 1.0/np.sqrt(self.effS), high = 1 - (1.0/np.sqrt(self.effS)), size=self.m)
        
    def score_update_instance(self,X_sample, index):
        X_normalized = (X_sample - self.minimum)/(self.maximum - self.minimum)
        X_normalized[np.abs(X_normalized) == np.inf] = 0
        
        score_instance = 0
        for r in range(self.m):
            Y = -1 * np.ones(len(self.V[r]))
            start_time = time.time()
            Y[range(len(self.V[r]))] = np.floor((X_normalized[np.array(self.V[r])] + np.array(self.alpha[r]))/float(self.f[r]))
            
            mod_entry = np.insert(Y, 0, r)
            mod_entry = tuple(mod_entry.astype(np.int))
            c = []
            for w in range(len(self.cmsketches)):
                try:
                    value = self.cmsketches[w][mod_entry]
                except KeyError, e:
                    value = (index, 0)
                    
                # Scoring the Instance    
                tstamp = value[0]
                wt = value[1]
                new_wt = wt * np.power(2, -self.decay * (index - tstamp))
                c.append(new_wt)
                
                # Update the instance
                new_tstamp = index
                self.cmsketches[w][mod_entry] = (new_tstamp, new_wt+1)
                
            min_c = min(c)
            c = np.log(1+min_c)
            if c < 0:
                print "Wrong here"
                print c
                print mod_entry
                print STOP
            score_instance = score_instance + c
            
        score = score_instance/self.m
        if(score < 0):
            print "SOME error @"
            print index
        if(np.isinf(score)):
            print score_instance, self.m
            print HEY
        return score

    def burn_in(self):
        # pp_data has the sample.
        # Normalize the data
        self._normalize()
        for r in range(self.m):
            for i in range(self.X_normalized.shape[0]):
                Y = -1 * np.ones(len(self.V[r]))
                Y[range(len(self.V[r]))] = np.floor((self.X_normalized[i,np.array(self.V[r])] + np.array(self.alpha[r]))/float(self.f[r]))
                
                mod_entry = np.insert(Y, 0, r)
                mod_entry = tuple(mod_entry.astype(np.int))
                
                for w in range(len(self.cmsketches)):
                    try:
                        value = self.cmsketches[w][mod_entry]
                    except KeyError, e:
                        value = (0, 0)
                    
                    #Setting Timestamp explicitly to 0
                    value = (0, value[1]+1)
                    self.cmsketches[w][mod_entry] = value
                    
        print "Processed:"+str(self.X_normalized.shape[0])+" points."
        print "Scoring...."
        
        scores=np.zeros(self.X_normalized.shape[0])
        for r in range(self.m):
            for i in range(self.X_normalized.shape[0]):
                Y = -1 * np.ones(len(self.V[r]))
                Y[range(len(self.V[r]))] = np.floor((self.X_normalized[i,np.array(self.V[r])] + np.array(self.alpha[r]))/float(self.f[r]))
                
                mod_entry = np.insert(Y, 0, r)
                mod_entry = tuple(mod_entry.astype(np.int))
                c = []
                for w in range(len(self.cmsketches)):
                    try:
                        value = self.cmsketches[w][mod_entry]
                    except KeyError, e:
                        print "Something is Wrong. This should not have happened"
                        print stop
                        
                    c.append(value[1])
                
                c = np.log2(min(c))
                scores[i]= scores[i] + c
            
        scores = scores/self.m
        return scores
        
def run_RSHash(X,y,sampling_points,decay):
    srhash = Stream_RS_Hash(X,y, sampling_points, decay)
    print "Burning In...."
    scores = list(srhash.burn_in())
    print "Burnt In..."
    start_time = time.time()
    for idx in tqdm(range(sampling_points,X.shape[0]), desc='Streaming Points...'):
        score = srhash.score_update_instance(X[idx], idx - sampling_points+1)
        if(idx%10000 == 0):
            print idx, time.time() - start_time
            start_time = time.time()
            
        scores.append(score)
        
    data = np.column_stack((scores,y))
    return data
    
        
def read_dataset2(in_file):
    data = np.loadtxt(in_file, delimiter=',')
    n,m = data.shape
    X = data[:,0:m-1]
    y = data[:,m-1]
    print n,m, X.shape, y.shape
    return X,y

#in_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/ODDS/DS/ionosphere_odds.txt"
#in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/ODDS/DS"
#out_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/ODDS/Results/Original/RSHash"
print "Running RSHash"
#file_name = sys.argv[1]
#num_runs = int(sys.argv[2])
#in_file = os.path.join(in_dir,file_name)
#out_file = os.path.join(out_dir, file_name)
in_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/Data/http_smtp_continuous_Shuffled.csv"
#out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/HttpSmtpContinuous/RSHash/1000_RSHash_015_Shuffled.csv"

#in_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/Data/http_smtp_continuous_Shuffled.csv"
#out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/HttpSmtpContinuous/RSHash_Streming_Results_Shuffled.csv"

#in_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/Data/smtp.csv"
#out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/SMTP/RSHash/256_RSHash_015.csv"

#in_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/Data/http_smtp_continuous.csv"
in_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/Data/SpamSmsCounts.csv"
out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/SpamSMSCounts/RSHash2/1Per_RSHash_015.csv"

X,y = read_dataset2(in_file)

out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/SpamSMSCounts/RSHash2/256_RSHash_0015.csv"
sampling_points=256
decay = 0.015
data = run_RSHash(X, y, sampling_points, decay)
np.savetxt(out_file, data, delimiter=",")

out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/SpamSMSCounts/RSHash2/1000_RSHash_0015.csv"
sampling_points=1000
decay = 0.015
data = run_RSHash(X, y, sampling_points, decay)
np.savetxt(out_file, data, delimiter=",")

out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/SpamSMSCounts/RSHash2/1Per_RSHash_0015.csv"
sampling_points=55
decay = 0.015
data = run_RSHash(X, y, sampling_points, decay)
np.savetxt(out_file, data, delimiter=",")

out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/SpamSMSCounts/RSHash2/10Per_RSHash_0015.csv"
sampling_points=557
decay = 0.015
data = run_RSHash(X, y, sampling_points, decay)
np.savetxt(out_file, data, delimiter=",")

out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/SpamSMSCounts/RSHash2/25Per_RSHash_0015.csv"
sampling_points=1393
decay = 0.015
data = run_RSHash(X, y, sampling_points, decay)
np.savetxt(out_file, data, delimiter=",")