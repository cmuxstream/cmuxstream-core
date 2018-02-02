import os
import sys
import numpy as np
import pandas as pd
import time
from scipy.io import loadmat,savemat
from scipy import sparse
from cryptography.hazmat.primitives.ciphers.modes import CTR

def read_dataset(in_file):
    data = np.loadtxt(in_file, delimiter=',')
    X = data[:,0:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    print X.shape, y.shape
    return X, y

def convert_to_HSTreeFile(data_dir, out_file):
    fw = open(out_file, "w")
    data_size = 2396130
    tot_cols = 3231962
    fw.write("*** PARAMETERS ***\t\t\t\t\tdistType\tDistance Measure\n")
    fw.write("dataSize:\t"+str(data_size)+"\t\t\t\t1\tCosineMeasure\n")
    fw.write("numClusters:\t"+str(2)+"\t\t\t\t2\tEuclideanMeasure\n")
    fw.write("distType:\t"+str(1)+"\t\t\t\t\t\n")
    fw.write("dimension:\t"+str(tot_cols)+"\t\t\t\t\t\n")
    fw.write("\n")
    fw.write("*** Data ***\n")
    
    
    #Read First Day
    cols_dict={}
    ctr=0
    f=open(os.path.join(data_dir,'Day0.svm'),'r')
    line=f.readline()
    while line:
        line=line.replace("\n","")
        split = line.split()
        label = int(split[0])
        if label==-1:
            label=0
        for i in range(1, len(split)):
            split2 = split[i].split(":")
            col_num=int(split2[0])
            val=float(split2[1])
            try:
                col_index=cols_dict[col_num]
            except Exception,e:
                cols_dict[col_num] = ctr
                ctr+=1
        line=f.readline()
    f.close()
    
    print "Num Cols Reduced to:"+str(len(cols_dict.keys()))
    for i in range(0, 121):
        print i
        f=open(os.path.join(data_dir,'Day'+str(i)+'.svm'),"r")
        line=f.readline()
        while line:
            line=line.replace("\n","")
            split = line.split()
            label = int(split[0])
            if label==-1:
                label=0
            fw.write(str(label))
            for i in range(1, len(split)):
                split2=split[i].split(":")
                col_num = int(split2[0])
                try:
                    col_index =  cols_dict[col_num]
                    fw.write("\t"+str(col_index)+":"+str(split2[1]))
                except Exception,e:
                    a=0
            fw.write("\n")
            line=f.readline()
        f.close()
        fw.flush()
    fw.close()
    
def daywise_convertsvmlight(folder, out_dir):
    for i in range(0,121):
        start_time = time.time()
        print "Processing Day:"+str(i)
        f=open(os.path.join(folder,"Day"+str(i)+".svm"),"r")
        out_file=os.path.join(out_dir,"Day"+str(i))
        #fw=open(out_file,'w')
        column_list = []
        data_vals = []
        labels = []
        line=f.readline()
        line_index=0
        while line:
            line_index+=1
            if(line_index%1000==0):
                print line_index
            cols=[]
            values=[]
            line=line.replace("\n","")
            split=line.split(" ")
            label = int(split[0])
            labels.append(label)
            dict_values={}
            for i in range(1,len(split)):
                split2 = split[i].split(":")
                cols.append(int(split2[0]))
                values.append(float(split2[1]))
                
            column_list.append(cols)
            data_vals.append(values)
            line=f.readline()
        f.close()
        print "Time Taken="+str(time.time()-start_time)
        lengths = [len(row) for row in column_list]
        cols = np.concatenate(column_list)
        rows = np.repeat(np.arange(len(column_list)), lengths)
        data_vals = np.concatenate(data_vals)
        m = sparse.coo_matrix((data_vals, (rows, cols)))
        print m.shape
        sparse.save_npz(out_file+".npz", m)
        np.save(out_file+"_Labels.npy",np.array(labels))
        savemat(out_file+".mat",{'vect':m})
        savemat(out_file+"_Labels.mat",{'labels':np.array(labels)})
        

def convert_svmlight_to_dense(folder,out_file):
    for i in range(0,121):
        start_time=time.time()
        data_file = os.path.join(folder,"Day"+str(i)+".npz")
        label_file = os.path.join(folder,"Day"+str(i)+"_Labels.npy")
        X = sparse.load_npz(data_file)
        X = sparse.csr_matrix(X)
        y = np.load(label_file)
        
        X = X.todense()
        np.savetxt(os.path.join(out_dir,"Day"+str(i)+".txt"),X,fmt='%.1f')
        print "Time taken="+str(time.time() - start_time)
    
def convert_svmlight_to_file(folder,out_file):
    fw=open(out_file,'w')
    column_list = []
    data_vals = []
    labels = []

    for i in range(0,121):
        start_time=time.time()
        print "Processing Day:"+str(i)
        f=open(os.path.join(folder,"Day"+str(i)+".svm"),"r")
        line=f.readline()
        line_index=0
        while line:
            line_index+=1
            if(line_index%1000==0):
                print line_index
            cols=[]
            values=[]
            line=line.replace("\n","")
            split=line.split(" ")
            label = int(split[0])
            if label == -1:
                label = 0
            labels.append(label)
            dict_values={}
            
            for i in range(1,len(split)):
                split2 = split[i].split(":")
                cols.append(int(split2[0]))
                values.append(float(split2[1]))
            
            column_list.append(cols)
            data_vals.append(values)
            line=f.readline()
        f.close()
        print "Time Taken="+str(time.time()-start_time)
    lengths = [len(row) for row in column_list]
    cols = np.concatenate(column_list)
    rows = np.repeat(np.arange(len(column_list)), lengths)
    data_vals = np.concatenate(data_vals)
    m = sparse.coo_matrix((data_vals, (rows, cols)))
    print m.shape
    sparse.save_npz(out_file+".npz", m)
    np.save(out_file+"_Labels.npy",np.array(labels))
    savemat(out_file+".mat",{'vect':m})
    savemat(out_file+"_Labels.mat",{'labels':np.array(labels)})
    

def modify_fraction(X,y,fraction):
    fraction_anoms = int(fraction * X.shape[0])
    anom_indexes = np.where(y==1)[0]
    norm_indexes = np.where(y==0)[0]
    selected_anoms = np.random.choice(anom_indexes,fraction_anoms,replace=False)
        
    X_mod = np.concatenate((X[norm_indexes,:],X[selected_anoms,:]), axis=0)
    y_mod = np.concatenate((y[norm_indexes],y[selected_anoms]), axis=0)
    
    data = np.column_stack((X_mod,y_mod))
    return data

def create_datasets(low_dir,high_dir,fraction,out_dir):
    low_ds_names = ["breast-cancer-wisconsin","magic-telescope","ionosphere","pima-indians"]
    high_ds_names = ["gisette","isolet","letter-recognition","madelon"]
    
    for ds_name in low_ds_names:
        print "Doing for:"+str(ds_name)
        X,y = read_dataset(os.path.join(low_dir, ds_name+"_overall.txt"))
        data = modify_fraction(X,y,fraction)
        np.savetxt(os.path.join(out_dir,ds_name+"_sampled.txt"), data, delimiter=',')
        
    for ds_name in high_ds_names:
        print "Doing for:"+str(ds_name)
        X,y = read_dataset(os.path.join(high_dir, ds_name+"_overall.txt"))
        data = modify_fraction(X, y, fraction)
        np.savetxt(os.path.join(out_dir, ds_name+"_sampled.txt"), data, delimiter=',')
    
def convert_mat_to_txt(in_file, out_file):
    arr = loadmat(in_file)    
    data = np.column_stack((arr['X'],arr['y']))
    print data
    np.savetxt(out_file,data,delimiter=',')
    
def convert_spectf_to_txt(in_file, out_file):
    X = np.loadtxt(in_file, delimiter = ",")
    labels = 1 - X[:,0]
    X = X[:,1:X.shape[1]]
    num_anoms = np.sum(labels)
    print X.shape[0],X.shape[1],num_anoms
    data = np.column_stack((X, labels))
    np.savetxt(out_file, data, delimiter=',')    
    
def main():
    BASE_DIR = "../../../New_Benchmark_Datasets"
    low_dir = os.path.join(BASE_DIR,"LowDim")
    high_dir = os.path.join(BASE_DIR,"HighDim")
    out_dir = os.path.join(BASE_DIR,"Overall_Dim")
    fraction = 0.05
    #create_datasets(low_dir,high_dir,fraction,out_dir)
    
    #convert_mat_to_txt(os.path.join(BASE_DIR,"ODDS/Orig/pima.mat"), 
    #                   os.path.join(BASE_DIR,"ODDS/New_DS/pima_odds.txt"))
    BASE_DIR = "/Users/hemanklamba/Documents/Experiments/Interactive_Outliers/LODA_Datasets/ODDS_Datasets"
    in_file = os.path.join(BASE_DIR, "SPECTF.test")
    out_file = os.path.join(BASE_DIR, "spectf_odds.txt")
    #convert_spectf_to_txt(in_file, out_file)
    
    folder = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/Data/url_svmlight"
    out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/Data/SPAM_URL.ssv"
    convert_svmlight_to_file(folder,out_file)
    
    folder = "../../../Data/url_svmlight"
    out_dir = "../../../Data/mod_url_svmlight"
    #daywise_convertsvmlight(folder, out_dir)
    #convert_svmlight_to_dense("../../../Data/mod_url_svmlight","../../../Data/mod_url_svmlight2")
    
    folder = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/Data/url_svmlight"
    out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/Data/parametersNdata_SpamURL.txt"
    #convert_to_HSTreeFile(folder, out_file)
    
if __name__ == '__main__':
    main()