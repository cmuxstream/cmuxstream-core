import os
import sys
import pandas as pd
import numpy as np
from shutil import copyfile

def create_statistics(in_dir, map_file, out_file):
    ds_map={}
    f=open(map_file,"r")
    line=f.readline()
    while line:
        line=line.replace("\n","")
        if((not "benchmark creation" in line) and (not "Timing" in line)):
            print line
            split=line.split(",")
            start_index =  int(split[0])
            end_index = int(split[1])
            for i in range(start_index, end_index):
                ds_map[i] = [split[2],split[3],split[4],split[5]]
        line=f.readline()
    f.close()
    print len(ds_map)
    num_meta=6
    list_files = os.listdir(in_dir)
    fw=open(out_file,"w")
    fw.write("Index,ARate,NumAnom,NumNorm,AnomDiffMean,NormDiffMean,ClusterednessScore,IrrelevanceScore,OriginalPtDiff,OriginalCluster,OriginalIrrel,OriginalRate\n")
    for in_file in list_files:
        df = pd.read_csv(os.path.join(in_dir,in_file))
        index = in_file.replace("parkinsons_benchmark_","")
        index = index.replace(".csv","")
        size = len(df)
        n_anom = len(df[df['ground.truth']=='anomaly'])
        n_norm = size-n_anom
        diff_mean = np.mean(df['diff.score'])
        anom_diff_mean = np.mean(df[df['ground.truth']=='anomaly']['diff.score'])
        norm_diff_mean = np.mean(df[df['ground.truth']=='nominal']['diff.score'])
        anomaly_rate = float(n_anom)/size
        ir_num = float(ds_map[int(index)][2])
        orig_cols = np.ceil((len(df.columns)-num_meta)/(np.power(ir_num,2)))
        true_data = df.iloc[:,int(num_meta):int(num_meta+orig_cols)]
        tot_var = np.sum(np.var(true_data))
        anom_var = np.sum(np.var(true_data[df['ground.truth']=='anomaly']))
        norm_var = np.sum(np.var(true_data[df['ground.truth']=='nominal']))
        clusteredness_score = np.log(norm_var/anom_var)
        ir_var = np.log(np.sum(df.iloc[:,int(num_meta):len(df.columns)].var(axis=1))/tot_var)
        fw.write(str(index)+","+str(anomaly_rate)+","+str(n_anom)+","+str(n_norm)+","+str(anom_diff_mean)+","+str(norm_diff_mean)+","+str(clusteredness_score)+","+str(ir_var)+","+str(ds_map[int(index)][0])+","+str(ds_map[int(index)][1])+","+str(ds_map[int(index)][2])+","+str(ds_map[int(index)][3])+"\n")

    fw.close()

def sampling(in_meta_file):
    '''
    This code will read in the input meta file.
    And Sample 1 dataset from each PD Level, with clusterdness score
    in the following brackets[ < -1, -1 to 0.5, 0.5 to 0, 0 to 0.5, 0.5 to 1, >1]
    '''
    df=pd.read_csv(in_meta_file)
    df_pd_arr = []
    df_pd_arr.append(df[df.OriginalPtDiff=='pd-0'])
    df_pd_arr.append(df[df.OriginalPtDiff=='pd-1'])
    df_pd_arr.append(df[df.OriginalPtDiff=='pd-2'])
    df_pd_arr.append(df[df.OriginalPtDiff=='pd-3'])
    df_pd_arr.append(df[df.OriginalPtDiff=='pd-4'])
    df_pd_arr.append(df[df.OriginalPtDiff=='pd-5'])
    print len(df_pd_arr)
    
    all_sampled_ds=[]
    for i in range(len(df_pd_arr)):
        sampled_indexes=[]
        sample1 = df_pd_arr[i][df_pd_arr[i].ClusterednessScore<-1]
        sample2 = df_pd_arr[i][(df_pd_arr[i].ClusterednessScore>=-1) & (df_pd_arr[i].ClusterednessScore<-0.5)]
        sample3 = df_pd_arr[i][(df_pd_arr[i].ClusterednessScore>=-0.5) & (df_pd_arr[i].ClusterednessScore<0)]
        sample4 = df_pd_arr[i][(df_pd_arr[i].ClusterednessScore>=0) & (df_pd_arr[i].ClusterednessScore<0.5)]
        sample5 = df_pd_arr[i][(df_pd_arr[i].ClusterednessScore>=0.5) & (df_pd_arr[i].ClusterednessScore<1)]
        sample6 = df_pd_arr[i][(df_pd_arr[i].ClusterednessScore>=1.0)]
        
        if(len(sample1)>0):
            sampled_indexes.append(int(sample1.sample(n=1)['Index']))
        else:
            sampled_indexes.append(None)
        
        if(len(sample2)>0):
            sampled_indexes.append(int(sample2.sample(n=1)['Index']))
        else:
            sampled_indexes.append(None)
        
        if(len(sample3)>0):
            sampled_indexes.append(int(sample3.sample(n=1)['Index']))
        else:
            sampled_indexes.append(None)
            
        if(len(sample4)>0):
            sampled_indexes.append(int(sample4.sample(n=1)['Index']))
        else:
            sampled_indexes.append(None)
            
        if(len(sample5)>0):
            sampled_indexes.append(int(sample5.sample(n=1)['Index']))
        else:
            sampled_indexes.append(None)
            
        if(len(sample6)>0):
            sampled_indexes.append(int(sample6.sample(n=1)['Index']))
        else:
            sampled_indexes.append(None)
        
        all_sampled_ds.append(sampled_indexes)
    return all_sampled_ds
#create_statistics("../Benchmarking_Dataset/ABALONE/benchmarks/",
#                   "abalone_meta", "../Benchmarking_Dataset/ABALONE/benchmarking_meta_info.csv")

def copy(in_dir, all_sampled_ds, out_dir):
    ds_names=["HighSc","MedSc","LowSc","LowCl","MedCl","HighCl"]
    meta_info=[]
    for i in range(len(all_sampled_ds)):
        sampled_ds=all_sampled_ds[i]
        diff_info = "PD-"+str(i)
        for j in range(len(sampled_ds)):
            if(sampled_ds[j]!=None):
                info = str(sampled_ds[j])+"_"+str(diff_info)+"_"+str(ds_names[j])
                meta_info.append(info)
                index = "%04d" % sampled_ds[j]
                src=os.path.join(in_dir,"parkinsons_benchmark_"+str(index)+".csv")
                dst=os.path.join(out_dir,"parkinsons_benchmark_"+str(info)+".csv")
                
                print "Copying:"+str(src)+" to "+str(dst)
                copyfile(src, dst)
    
#in_meta_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/benchmarking_meta_info.csv"
#in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/abalone/benchmarks"
#out_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/abalone/consolidated"

in_dir = "../Benchmarking_Dataset/PARKINSONS/benchmarks"
map_file = "parkinsons_meta"
out_file = "../Benchmarking_Dataset/PARKINSONS/benchmarking_meta_info"
create_statistics(in_dir,map_file, out_file)

in_meta_file = "../Benchmarking_Dataset/PARKINSONS/benchmarking_meta_info"
#in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/abalone/benchmarks"
out_dir = "../Benchmarking_Dataset/PARKINSONS/consolidated"

all_sampled_ds=sampling(in_meta_file)
copy(in_dir, all_sampled_ds, out_dir)