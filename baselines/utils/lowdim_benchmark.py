import os
import sys
import numpy as np
import pandas as pd
from os.path import basename
from sklearn.metrics import pairwise_distances

def read_dataset(in_file):
    data = np.loadtxt(in_file, delimiter=',')
    X = data[:,0:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    return X, y

def distort_lowdim(X, noisy_column_percentage, noise_level):
    '''
        Input:
            X: Input dataset
            noisy_column_percentage: Number of noisy columns to be added = noisy_column_percentage * ncols(X)
            noise_level: Level of noise. Each noisy column is created with 
                        Gaussian( mu = noise_level * avg.mean(X), sigma = noise_level * avg.variance(X))
        Output:
            X_noisy: Output dataset
            
    '''
    n,m = X.shape
    num_noisy_columns = int((noisy_column_percentage * m)/100.0)
    #print "Number of noisy columns added="+str(num_noisy_columns)
    
    mu = np.mean(X)
    std = np.std(X)
    
    #print "Mean="+str(mu)+", Std="+str(std)
    noisy_mu = noise_level * mu
    noisy_var = noise_level * std
    #print "Noisy Mu="+str(noisy_mu)+", Noisy std="+str(noisy_std)
    
    #orig_distances = pairwise_distances(X_noisy, X_noisy)
    #print X_noisy.shape, np.mean(orig_distances)
    
    X_noise = np.random.normal(loc=noisy_mu, scale=noisy_var, size=(n,num_noisy_columns))
    X_noisy = np.concatenate((X, X_noise), axis=1)
        #orig_distances = pairwise_distances(X_noisy, X_noisy)
        #print "\t"+str(X_noisy.shape)+"\t"+str(np.mean(X_noisy))+"\t"+str(np.std(X_noisy))+"\t"+str(np.mean(orig_distances))
    return X_noisy

def main():
    in_file = sys.argv[1]
    noisy_column_percentage = float(sys.argv[2])
    noise_level = float(sys.argv[3])
    out_dir = sys.argv[4]
    
    X,y = read_dataset(in_file)
    X_noisy = distort_lowdim(X,noisy_column_percentage, noise_level)
    out_file = os.path.join(out_dir,basename(in_file)+"_"+str(noisy_column_percentage)+"_"+str(noise_level)+"_NOISY")
    data = np.column_stack((X_noisy,y))
    np.savetxt(out_file, data, delimiter=',')
    
if __name__ == '__main__':
    main()

    
    
     
