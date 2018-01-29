import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import svgwrite
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from _tkinter import create

def get_auc_ap(in_dir,out_file):
    fw=open(out_file,'w')
    list_files = os.listdir(in_dir)
    for in_file in list_files:
        if(".pkl" not in in_file):
            print "reading:"+str(in_file)
            row_name = in_file.replace(".txt","")
            row_name = row_name.replace("_NOISY","")
            f=open(os.path.join(in_dir,in_file),"r")
            lines=f.readlines()
            if len(lines)==0:
                continue
            line=lines[len(lines)-1]
            split=line.split(",")
            mean_auc = float(split[0])
            std_auc = float(split[1])
            mean_ap = float(split[2])
            std_ap = float(split[3])
            fw.write(str(row_name)+","+str.format('{0:.3f}', mean_auc)+","+str.format('{0:.3f}',std_auc)+","+str.format('{0:.3f}',mean_ap)+","+str.format('{0:.3f}',std_ap)+"\n")
    fw.close()
            
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
    
#isofor = analyze_result("../../../Results/Baseline_Results_Irrel/IForest_50.txt")
#loda = analyze_result("../../../Results/Baseline_Results_Irrel/LODA_50.txt")
#rshash = analyze_result("../../../Results/Baseline_Results_Irrel/RSHash_50.txt")
#hstrees = analyze_result("../../HighDim_Outliers/Benchmark_Datasets/Results/HSTrees_10.txt")
#res_arr = plotter(isofor, loda, rshash, hstrees)
#res_arr = plotter(isofor, loda, rshash)

#plot(res_arr[0], "Parkinsons", "../../../Results/Plots_Irrel")
#plot(res_arr[1], "Abalone", "../../../Results/Plots_Irrel")
#plot(res_arr[2], "Ecoli", "../../../Results/Plots_Irrel")

def get_ds_string(in_file,arrs):
    f=open(in_file,"r")
    delimiter=","
    line=f.readline()
    while line:
        line=line.replace("\n","")
        split=line.split(delimiter)
        ds_name = split[0]
        if ".tsv" not in in_file:
            mean_auc = float(split[1])
            std_auc = float(split[2])
            mean_ap = float(split[3])
            std_ap = float(split[4])
        else:
            mean_ap=float(split[1])
            std_ap=float(split[2])
            mean_auc=float(split[3])
            std_auc=float(split[4])
            
        temp_str = " $"+str(mean_ap)+" \\pm "+str(std_ap)+"$"
        if ds_name in arrs:
            temp = arrs[ds_name]
            temp = temp+" & "+temp_str
        else:
            temp = temp_str
        
        arrs[ds_name]= temp 
        line=f.readline()
    f.close()
    return arrs

def get_latex_table(in_dir):
    arrs={}
    arrs = get_ds_string(os.path.join(in_dir,"iforest.txt"), arrs)
    arrs = get_ds_string(os.path.join(in_dir,"hstrees.txt"), arrs)
    arrs = get_ds_string(os.path.join(in_dir,"rshash.txt"), arrs)
    arrs = get_ds_string(os.path.join(in_dir,"loda.txt"), arrs)
    arrs = get_ds_string(os.path.join(in_dir,"xstream_lowdim.tsv"),arrs)
    
    for key in sorted(arrs.iterkeys()):
        split = key.split("_")
        if len(split)>2:
            #name = split[0]+" ("+split[2]+","+split[3]+","+split[4]+" )"
            name = split[0]+" ("+split[2]+","+split[3]+" )"
        else:
            name = split[0]
        print name+"&"+arrs[key]+"    \\\\"
    
def get_result_array(in_file):
    stat_arr=[]
    ds_arr=[]
    f=open(in_file,"r")
    line=f.readline()
    while line:
        line=line.replace("\n","")
        delimiter = ","
        split=line.split(delimiter)
        ds_name = split[0]
        ds_arr.append(ds_name)
        if ".tsv" in in_file:
            stat_arr.append(float(split[1]))
        else:
            stat_arr.append(float(split[3]))
        line=f.readline()
    return stat_arr,ds_arr
    
def get_file_for_hypTesting2(res_dir, out_dir,name):
    fw_ds = open(os.path.join(out_dir,name+".csv"),'w')
    ifor_low_arr,ifor_ds = get_result_array(os.path.join(res_dir,"iforest.txt"))
    hst_low_arr,hst_ds = get_result_array(os.path.join(res_dir,"hstrees.txt"))
    rsh_low_arr,rsh_ds = get_result_array(os.path.join(res_dir,"rshash.txt"))
    loda_low_arr,loda_ds = get_result_array(os.path.join(res_dir,"loda.txt"))
    xst_low_arr,xst_ds = get_result_array(os.path.join(res_dir,"xstream_lowdim.tsv"))
    
    ds_names = [x.replace("_sampled","") for x in ifor_ds]
    print ds_names
    #fw_ds.write((",".join(ds for ds in ifor_ds)+"\n"))
    fw_ds.write(",".join(format(x, "0.4f") for x in ifor_low_arr)+"\n")
    fw_ds.write(",".join(format(x, "0.4f") for x in hst_low_arr)+"\n")
    fw_ds.write(",".join(format(x, "0.4f") for x in rsh_low_arr)+"\n")
    fw_ds.write(",".join(format(x, "0.4f") for x in loda_low_arr)+"\n")
    fw_ds.write(",".join(format(x, "0.4f") for x in xst_low_arr)+"\n")
    fw_ds.close()
    return ds_names
    
def get_file_for_hypTesting(lowdim_dir,highdim_dir,out_dir):
    fw_low = open(os.path.join(out_dir,"LowDim.csv"),'w')
    fw_high = open(os.path.join(out_dir,"HighDim.csv"),'w')
    fw_overall =  open(os.path.join(out_dir,"Overall.csv"),'w')

    #Open lowDim
    ifor_low_arr,ifor_ds = get_result_array(os.path.join(lowdim_dir,"iforest.txt"))
    hst_low_arr,hst_ds = get_result_array(os.path.join(lowdim_dir,"hstrees.txt"))
    rsh_low_arr,rsh_ds = get_result_array(os.path.join(lowdim_dir,"rshash.txt"))
    loda_low_arr,loda_ds = get_result_array(os.path.join(lowdim_dir,"loda.txt"))
    xst_low_arr,xst_ds = get_result_array(os.path.join(out_dir,"xstream_lowdim.tsv"))
    
    fw_low.write(",".join(format(x, "0.4f") for x in ifor_low_arr)+"\n")
    fw_low.write(",".join(format(x, "0.4f") for x in hst_low_arr)+"\n")
    fw_low.write(",".join(format(x, "0.4f") for x in rsh_low_arr)+"\n")
    fw_low.write(",".join(format(x, "0.4f") for x in loda_low_arr)+"\n")
    fw_low.write(",".join(format(x, "0.4f") for x in xst_low_arr)+"\n")
    
    #Open HighDim
    ifor_high_arr,ifor_ds = get_result_array(os.path.join(highdim_dir,"iforest.txt"))
    hst_high_arr,hst_ds = get_result_array(os.path.join(highdim_dir,"hstrees.txt"))
    rsh_high_arr,rsh_ds = get_result_array(os.path.join(highdim_dir,"rshash.txt"))
    loda_high_arr,loda_ds = get_result_array(os.path.join(highdim_dir,"loda.txt"))
    xst_high_arr,xst_ds = get_result_array(os.path.join(out_dir,"xstream_highdim.tsv"))
    
    fw_high.write(",".join(format(x, "0.4f") for x in ifor_high_arr)+"\n")
    fw_high.write(",".join(format(x, "0.4f") for x in hst_high_arr)+"\n")
    fw_high.write(",".join(format(x, "0.4f") for x in rsh_high_arr)+"\n")
    fw_high.write(",".join(format(x, "0.4f") for x in loda_high_arr)+"\n")
    fw_high.write(",".join(format(x, "0.4f") for x in xst_high_arr)+"\n")
    
    #combine both
    ifor_arr = ifor_low_arr+ifor_high_arr
    rsh_arr = rsh_low_arr+rsh_high_arr
    loda_arr = loda_low_arr+loda_high_arr
    hst_arr = hst_low_arr+hst_high_arr
    xst_arr = xst_low_arr+xst_high_arr
    
    fw_overall.write(",".join(format(x, "0.4f") for x in ifor_arr)+"\n")
    fw_overall.write(",".join(format(x, "0.4f") for x in hst_arr)+"\n")
    fw_overall.write(",".join(format(x, "0.4f") for x in rsh_arr)+"\n")
    fw_overall.write(",".join(format(x, "0.4f") for x in loda_arr)+"\n")
    fw_overall.write(",".join(format(x, "0.4f") for x in xst_arr)+"\n")
    
    fw_overall.close()
    fw_low.close()
    fw_high.close()
    
def compute_avg_ranks_overall(in_file,ds_names):
    ds_arrs = {}
    arr = np.loadtxt(in_file,delimiter=",")
    #orig_arr =  arr[:,np.arange(0,arr.shape[1],5)]
    orig_arr =  arr
    method_names = ["IF","HST","RSH","LODA","XS"]
    #ideal_ds_names = ["cancer","ionosphere","telescope","indians","gisette","isolet","letter","madelon"]
    
    ranks_arr =  np.empty(orig_arr.shape)
    for dataset in range(orig_arr.shape[1]):
        ranks = rankdata(-orig_arr[:,dataset])
        ranks_arr[:,dataset] = ranks           
        
    text = ""
    print "\t".join(method_names)
    
    for i in range(orig_arr.shape[1]):
        ds_name = ds_names[i]
        text = text + str(ds_name)+"\t"
        for j in range(len(method_names)):
            text = text + str(orig_arr[j][i])+"("+str(ranks_arr[j][i])+")\t"
            
        text = text+"\n"
        
    print text
    avg_ranks = np.mean(ranks_arr,axis=1)
    print avg_ranks
    return avg_ranks
    
def compute_avg_ranks_low(in_file):
    arr = np.loadtxt(in_file,delimiter=",")
    orig_arr = arr
    #orig_arr =  arr[:,np.arange(0,arr.shape[1],5)]
    method_names = ["IF","HST","RSH","LODA","XS"]
    ds_names = ["cancer100","cancer1000","cancer2000","cancer5000","ionosphere100","ionosphere1000","ionosphere2000","ionosphere5000","telescope100","telescope1000","telescope2000","telescope5000","indians100","indians1000","indians2000","indians5000"]
    
    print orig_arr.shape
    ranks_arr = np.empty(orig_arr.shape)
    
    for dataset in range(orig_arr.shape[1]):
        temp = []
        ranks = rankdata(-orig_arr[:,dataset])
        ranks_arr[:,dataset] = ranks
    
    print ranks_arr
    print HEY
        
    text = ""
    print "\t".join(method_names)
    for i in range(len(sel_cols)):
        ds_name = ds_names[sel_cols[i]]
        text = text + str(ds_name)+"\t"
        for j in range(len(method_names)):
            text = text + str(orig_arr[j][sel_cols[i]])+"("+str(ranks_arr[j][i])+")\t"
            
        text = text+"\n"
        
    print text
    avg_ranks = np.mean(ranks_arr,axis=1)
    print avg_ranks
    return avg_ranks
            
def find_crit_differences_nemeyeni(avg_ranks,N):
    q5 = 2.728
    k = avg_ranks.shape[0]
    cd = q5 * np.sqrt((k * (k+1))/(6.0*N))
    print "critical difference="+str(cd)
    
    critical_lines = []
    for i in range(avg_ranks.shape[0]):
        for j in range(avg_ranks.shape[0]):
            if(i!=j):
                diff = avg_ranks[i] - avg_ranks[j]
                if diff > cd:
                    print str(i+1)+" is significant than "+str(j+1)
                    critical_lines.append([i+1,j+1])        

    scores = {}
    scores["IF"] = avg_ranks[0]
    scores["HST"] = avg_ranks[1]
    scores["RSH"] = avg_ranks[2]
    scores["LODA"] = avg_ranks[3]
    scores["XS"] = avg_ranks[4]
    
    return scores, critical_lines

def read_dataset2(filename):
    data = np.loadtxt(filename, delimiter=',')
    n,m = data.shape
    X = data[:,0:m-1]
    y = data[:,m-1]
    print n,m, X.shape, y.shape
    return X,y

def create_latex_for_tables(in_dir):
    in_files = os.listdir(in_dir)
    for in_file in in_files:
        if("NOISY" not in in_file):
            X,y = read_dataset2(os.path.join(in_dir,in_file))
            in_file = in_file.replace(".txt","")
            print pd.value_counts(y)
            print in_file+" & "+str(X.shape[0])+" & "+str(X.shape[1])+" & "+str(np.sum(y))+" \\\\"
            
        
isofor_dir = "../../../New_Benchmark_Datasets/ODDS/Results/LowDim_Noise/HSTrees"
iso_file = "../../../New_Benchmark_Datasets/ODDS/Results/LowDim_Noise/hstrees.txt"
#get_auc_ap(isofor_dir, iso_file)
#print HEY
#create_latex_for_tables("../../../New_Benchmark_Datasets/Overall_Dim")
create_latex_for_tables("../../../../Interactive_Outliers/LODA_Datasets/ODDS_Datasets/")
print HEY
#get_latex_table("../../../New_Benchmark_Datasets/ODDS/Results/LowDim_Noise")

lowdim_dir = "../../../New_Benchmark_Datasets/Results_Static/LowDim"
highdim_dir = "../../../New_Benchmark_Datasets/Results_Static/HighDim_Option2"
out_dir = "../../../New_Benchmark_Datasets/Results_Static/HypTesting"
#get_file_for_hypTesting(lowdim_dir, highdim_dir, out_dir)

res_dir = "../../../New_Benchmark_Datasets/ODDS/Results/LowDim_Noise"
out_dir = "../../../New_Benchmark_Datasets/ODDS/Results/HypTesting"
ds_names = get_file_for_hypTesting2(res_dir, out_dir,name="LowDim_16")

in_file = "../../../New_Benchmark_Datasets/ODDS/Results/HypTesting/LowDim_16.csv"
#avg_ranks = compute_avg_ranks_overall(in_file, ds_names)
##avg_ranks = compute_avg_ranks_overall(in_file)
#scores, critical_lines = find_crit_differences_nemeyeni(avg_ranks,N=16)
#plot_result(scores, critical_lines)

in_dir = "../../../New_Benchmark_Datasets/ODDS/DS"
#create_latex_for_tables(in_dir)