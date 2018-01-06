import os
import sys
import numpy as np

def read_dataset(filename):
    data = np.loadtxt(filename, delimiter=',')
    n,m = data.shape
    X = data[:,0:m-1]
    y = data[:,m-1]
    print n,m, X.shape, y.shape
    return X,y

def chunks(l,n):
    """
    Creates chunks of size n in list l
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]
    
    
def shuffle_dataset(X,y, mode, params):
    """
    Shuffles the dataset, so that it works for 'Streaming' case.
    mode:
        if mode == "Random", shuffle the data randomly and output.
        elif mode == "Clustering", then insert anomalies clustered in time.
        
    params:
        this is mainly for clustering mode.
            num_points = Number of anomalies to be inserted at each time.
    """
    
    if mode == "Random":
        arr =  range(X.shape[0])
        np.random.shuffle(arr)
        X_shuffled = X[arr]
        y_shuffled = y[arr]
        
    elif mode == "Clustering":
        X_shuffled = []
        y_shuffled = []
        anomaly_indexes = np.where(y==1)[0]
        normal_indexes =  np.where(y==0)[0]
        np.random.shuffle(anomaly_indexes)
        
        anom_rate = float(len(anomaly_indexes))/X.shape[0]
        anom_chunks = list(chunks(anomaly_indexes,params['n']))
        
        print len(anom_chunks)
        for i in range(len(anom_chunks)):
            normal_indexes = np.append(normal_indexes, [-1])
    
        curr_anom_chunk=0
        np.random.shuffle(normal_indexes)
        for index in normal_indexes:
            if index == -1:
                for j in range(len(anom_chunks[curr_anom_chunk])):
                    X_shuffled.append(X[anom_chunks[curr_anom_chunk][j]])
                    y_shuffled.append(y[anom_chunks[curr_anom_chunk][j]])
            else:
                X_shuffled.append(X[index])
                y_shuffled.append(y[index])
                
        X_shuffled = np.array(X_shuffled)
        y_shuffled = np.array(y_shuffled)   
    assert np.sum(y) == np.sum(y_shuffled)
    
    return X_shuffled, y_shuffled
    
def create_datasets(ds_name, mode, num_ds, params, out_dir):
    in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/HighDim/New2"
    ds_file = os.path.join(in_dir, ds_name)
    X, y = read_dataset(ds_file)
    
    for i in range(num_ds):
        print ds_name+"--------"+str(i)
        X_new, y_new = shuffle_dataset(X,y, mode, params)
        out_file = os.path.join(out_dir,ds_name+"_"+mode+"_"+str(i))
        data = np.column_stack((X_new,y_new))
        np.savetxt(out_file, data, delimiter=',')
    
    
if __name__ == '__main__':
    out_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/Streaming_HighDim"
    ds_name = sys.argv[1]
    mode = sys.argv[2]
    num_ds = int(sys.argv[3])
    params = {}
    params['n'] = 10
    create_datasets(ds_name, mode, num_ds, params, out_dir)
    
    
    
    
    