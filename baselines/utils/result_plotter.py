import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from notebook._tz import isoformat

def analyze_result(in_file):
    res_arr=[]
    f=open(in_file,"r")
    line=f.readline()
    while line:
        line=line.replace("\n","")
        split=line.split(",")
        name = split[0]
        mean_auc = float(split[1])
        std_auc = float(split[2])
        mean_ap = float(split[3])
        std_ap = float(split[4])
        res_arr.append([name, mean_auc, std_auc, mean_ap, std_ap])
        line=f.readline()
    return res_arr
    
def plotter(isofor, loda, rshash):
    parkinsons_loda=[]
    parkinsons_isofor=[]
    parkinsons_hstrees=[]
    parkinsons_rshash=[]
    abalone_loda=[]
    abalone_isofor=[]
    abalone_hstrees=[]
    abalone_rshash=[]
    ecoli_loda=[]
    ecoli_isofor=[]
    ecoli_hstrees=[]
    ecoli_rshash=[]
    

    fi_levels={}
    fi_levels['HighFI']=2.0
    fi_levels['MedFI']=1.5
    fi_levels['LowFi']=1.2
    fi_levels['None']=1.0
    
    for i in range(len(isofor)):
        name = isofor[i][0]
        name = name.replace(".csv","")
        split = name.split("_")
        ds_name = split[0]
        diff = split[3]
        diff = int(diff.split("-")[1])
        irrel = split[4]
        if(irrel=="Regular"):
            irrel=1.0
        else:
            irrel = float(split[4].split("=")[1])
        irrel_level = irrel
        
        if("parkinsons" in name):
            parkinsons_isofor.append(np.array([diff,irrel_level,isofor[i][1],isofor[i][2],isofor[i][3],isofor[i][4]]))
        elif("abalone" in name):
            abalone_isofor.append([diff,irrel_level,isofor[i][1],isofor[i][2],isofor[i][3],isofor[i][4]])
        else:
            ecoli_isofor.append([diff,irrel_level,isofor[i][1],isofor[i][2],isofor[i][3],isofor[i][4]])

    for i in range(len(loda)):
        name = loda[i][0]
        name = name.replace(".csv","")
        split = name.split("_")
        ds_name = split[0]
        diff = split[3]
        diff = int(diff.split("-")[1])
        irrel = split[4]
        if(irrel=="Regular"):
            irrel=1.0
        else:
            irrel = float(split[4].split("=")[1])
        irrel_level = irrel
        
        if("parkinsons" in name):
            parkinsons_loda.append(np.array([diff,irrel_level,loda[i][1],loda[i][2],loda[i][3],loda[i][4]]))
        elif("abalone" in name):
            abalone_loda.append([diff,irrel_level,loda[i][1],loda[i][2],loda[i][3],loda[i][4]])
        else:
            ecoli_loda.append([diff,irrel_level,loda[i][1],loda[i][2],loda[i][3],loda[i][4]])
            
    for i in range(len(rshash)):
        name = rshash[i][0]
        name = name.replace(".csv","")
        split = name.split("_")
        ds_name = split[0]
        diff = split[3]
        diff = int(diff.split("-")[1])
        irrel = split[4]
        if(irrel=="Regular"):
            irrel=1.0
        else:
            irrel = float(split[4].split("=")[1])
        irrel_level = irrel
        
        if("parkinsons" in name):
            parkinsons_rshash.append(np.array([diff,irrel_level,rshash[i][1],rshash[i][2],rshash[i][3],rshash[i][4]]))
        elif("abalone" in name):
            abalone_rshash.append([diff,irrel_level,rshash[i][1],rshash[i][2],rshash[i][3],rshash[i][4]])
        else:
            ecoli_rshash.append([diff,irrel_level,rshash[i][1],rshash[i][2],rshash[i][3],rshash[i][4]])
    
    '''        
    for i in range(len(hstrees)):
        name = hstrees[i][0]
        name = name.replace(".csv","")
        split = name.split("_")
        ds_name = split[0]
        diff = split[3]
        diff = int(diff.split("-")[1])
        irrel = split[4]
        irrel = float(split[4].split("=")[1])
        irrel_level = fi_levels[irrel]
        
        if("parkinsons" in name):
            parkinsons_hstrees.append(np.array([diff,irrel_level,loda[i][1],loda[i][2],loda[i][3],loda[i][4]]))
        elif("abalone" in name):
            abalone_hstrees.append([diff,irrel_level,loda[i][1],loda[i][2],loda[i][3],loda[i][4]])
        else:
            ecoli_hstrees.append([diff,irrel_level,loda[i][1],loda[i][2],loda[i][3],loda[i][4]])
    '''
        
    parkinsons_isofor=np.array(parkinsons_isofor)
    abalone_isofor=np.array(abalone_isofor)
    ecoli_isofor=np.array(ecoli_isofor)
    parkinsons_loda=np.array(parkinsons_loda)
    abalone_loda=np.array(abalone_loda)
    ecoli_loda=np.array(ecoli_loda)
    parkinsons_rshash=np.array(parkinsons_rshash)
    abalone_rshash=np.array(abalone_rshash)
    ecoli_rshash=np.array(ecoli_rshash)
    #parkinsons_hstrees=np.array(parkinsons_hstrees)
    #abalone_hstrees=np.array(abalone_hstrees)
    #ecoli_hstrees=np.array(ecoli_hstrees)
    
    res_arr = []
    #res_arr.append([parkinsons_isofor,parkinsosn_loda,parkinsons_rshash, parkinsons_hstrees])
    res_arr.append([parkinsons_isofor,parkinsons_loda,parkinsons_rshash])
    #res_arr.append([abalone_isofor,abalone_loda,abalone_rshash, abalone_hstrees])
    res_arr.append([abalone_isofor,abalone_loda,abalone_rshash])
    #res_arr.append([ecoli_isofor,ecoli_loda, ecoli_rshash,ecoli_hstrees])
    res_arr.append([ecoli_isofor,ecoli_loda, ecoli_rshash])
    
    return res_arr

def plot(arr, name, out_dir):
    isofor=arr[0]
    loda=arr[1]
    rshash=arr[2]
    
    index=0
    x_arr=[]
    y1_arr=[]
    y2_arr=[]
    y3_arr=[]
    x_ticks=[]
    for diff_level in range(5):
        for cl_levels in [1.0, 1.2, 1.5, 2.0]:
            temp1 = isofor[(isofor[:,0]==diff_level) & (isofor[:,1]==cl_levels)]
            temp2 = loda[(loda[:,0]==diff_level) & (loda[:,1]==cl_levels)]
            temp3 = rshash[(rshash[:,0]==diff_level) & (rshash[:,1]==cl_levels)]
            if((len(temp1)>0) and (len(temp2)>0) and (len(temp3)>0)):
                x_ticks.append("D-"+str(diff_level)+"_FI-"+str(cl_levels))
                index+=1
                x_arr.append(index)
                y1_arr.append(temp1)
                y2_arr.append(temp2)
                y3_arr.append(temp3)

    y1_arr=np.array(y1_arr)
    y2_arr=np.array(y2_arr)
    y3_arr=np.array(y3_arr)
    x_arr=np.array(x_arr)
    plt.figure(figsize=(12,10))
    plt.bar(x_arr-0.3,y1_arr[:,0,2],width=0.2,yerr=y1_arr[:,0,3])
    plt.bar(x_arr+0.0,y2_arr[:,0,2],width=0.2,yerr=y2_arr[:,0,3])
    plt.bar(x_arr+0.3,y3_arr[:,0,2],width=0.2,yerr=y3_arr[:,0,3])
    plt.xticks(x_arr,x_ticks,rotation=70)
    plt.legend(['IsoForest','LODA', 'RSHash'])
    plt.title("Mean AUC-"+str(name))
    plt.savefig(os.path.join(out_dir,name)+"_AUC.png")
    plt.close()
    
    plt.figure(figsize=(12,10))
    plt.bar(x_arr-0.3,y1_arr[:,0,4],width=0.2,yerr=y1_arr[:,0,5])
    plt.bar(x_arr+0.0,y2_arr[:,0,4],width=0.2,yerr=y2_arr[:,0,5])
    plt.bar(x_arr+0.3,y3_arr[:,0,4],width=0.2,yerr=y3_arr[:,0,5])
    plt.xticks(x_arr,x_ticks,rotation=70)
    plt.legend(['IsoForest','LODA', 'RSHash'])
    plt.title("Mean AP")
    plt.savefig(os.path.join(out_dir,name)+"_AP.png")
    plt.close()
    
isofor = analyze_result("../../../Results/Baseline_Results_Irrel/IForest_50.txt")
loda = analyze_result("../../../Results/Baseline_Results_Irrel/LODA_50.txt")
rshash = analyze_result("../../../Results/Baseline_Results_Irrel/RSHash_50.txt")
#hstrees = analyze_result("../../HighDim_Outliers/Benchmark_Datasets/Results/HSTrees_10.txt")
#res_arr = plotter(isofor, loda, rshash, hstrees)
res_arr = plotter(isofor, loda, rshash)

plot(res_arr[0], "Parkinsons", "../../../Results/Plots_Irrel")
plot(res_arr[1], "Abalone", "../../../Results/Plots_Irrel")
plot(res_arr[2], "Ecoli", "../../../Results/Plots_Irrel")


