import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, scale
from scipy.io import loadmat
from HSTrees import HSTrees

def convert_to_HSTreeFile(in_file, out_file):
    fw = open(out_file, "w")
    data = np.loadtxt(in_file, delimiter = "\t")
    X = data[:,0:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    fw.write("*** PARAMETERS ***\t\t\t\t\tdistType\tDistance Measure\n")
    fw.write("dataSize:\t"+str(X.shape[0])+"\t\t\t\t1\tCosineMeasure\n")
    fw.write("numClusters:\t"+str(2)+"\t\t\t\t2\tEuclideanMeasure\n")
    fw.write("distType:\t"+str(1)+"\t\t\t\t\t\n")
    fw.write("dimension:\t"+str(X.shape[1])+"\t\t\t\t\t\n")
    fw.write("\n")
    fw.write("*** Data ***\n")
    data=np.column_stack((y,X))
    np.savetxt(fw, data, delimiter='\t',fmt='%.4f')
    fw.close()
    
    
in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/Data"
num_runs = 1
#in_file = os.path.join(in_dir,"http_smtp_continuous.csv")
#out_file = os.path.join(in_dir,"parametersNdata_HttpSmtpContinous.txt")
in_file = os.path.join(in_dir,"http_smtp_continuous_Shuffled.csv")
out_file = os.path.join(in_dir,"parametersNdata_HttpSmtpContinous_Shuffled.txt")
#out_file = os.path.join("", file_name)
in_file = os.path.join(in_dir,"spam-sms-preprocessed-counts.tsv")
out_file = os.path.join(in_dir,"parametersNdata_SpamSmsCounts.txt")
convert_to_HSTreeFile(in_file, out_file)