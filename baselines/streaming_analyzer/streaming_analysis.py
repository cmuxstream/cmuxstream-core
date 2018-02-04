import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy import sparse

def compute_statistics(scores, labels):
    avg_precision = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)
    return auc, avg_precision

def convert_from_HSTreeFile(in_file, out_file):
    fw = open(out_file, "w")
    f = open(in_file, "r")
    X = []
    y = []
    data_start=False
    line=f.readline()
    while line:
        if(data_start):
            line=line.replace("\n","")
            line=line.replace("\r","")
            split = line.split("\t")
            label = split[0]
            try:
                X.append(np.array(split[1:len(split)],dtype='float'))
            except Exception, e:
                print split
            y.append(int(float(label)))
        
        if("*** Data ***" in line):
            data_start=True
        line=f.readline()

    X=np.array(X)
    y=np.array(y)
    data=np.column_stack((X,y))
    np.savetxt(fw, data, delimiter=',',fmt='%.4f')
    fw.close()

def Analyze_Stream_HSTree(score_file):
    auc_arr=[]
    ap_arr=[]
    f=open(score_file,"r")
    line=f.readline()
    scores = []
    labels = []
    min_window_size = 256
    while line:
        line=line.replace("\n","")
        split=line.split(",")
        label=int(split[0])
        score=float(split[1])
        scores.append(-score)
        labels.append(label)
        line=f.readline()
    f.close()
    
    x_arr=[]
    for x in np.arange(min_window_size,len(scores),1000):
        auc,ap = compute_statistics(scores[0:x], labels[0:x])
        if not np.isnan(ap):
            auc_arr.append(auc)
            ap_arr.append(ap)
            x_arr.append(x)
    
    x_arr=np.array(x_arr)
    plt.figure(figsize=(8,6))
    plt.plot(range(len(ap_arr)),ap_arr, linewidth=5.0)
    plt.xticks(np.arange(0,len(ap_arr),10000),x_arr[np.arange(0,len(x_arr),10000)],rotation=70)
    plt.xlabel("Number of Points Seen", fontsize=20)
    plt.ylabel("Running AP", fontsize=20)
    plt.savefig(score_file+".pdf")

def Analyze_Stream_Dir(score_dir,method):
    fw=open(os.path.join(score_dir,"AUC_AP_Stats.txt"),"w")
    list_files = os.listdir(score_dir)
    for in_file in list_files:
        if ((method == "HST") & ("AnormalyScore" in in_file) & (".csv" in in_file)):
            fname = in_file.replace(".csv","")
            print "Reading"+str(fname)
            auc, ap = Analyze_Stream_Data(score_dir, fname, method)
            fw.write(fname+"\t"+str(auc)+"\t"+str(ap)+"\n")
        elif(method!="HST"):
            if(".csv" in in_file):
                fname = in_file.replace(".csv","")
                auc, ap = Analyze_Stream_Data(score_dir, fname,method)
                fw.write(fname+"\t"+str(auc)+"\t"+str(ap)+"\n")
    fw.close()

def Analyze_Stream_SpamURL(scores, labels,out_file):
    nelements = len(scores)
    print np.sum(labels)
    anom_index = np.where(labels==1.0)[0][0]
    print "Anom Index="+str(anom_index)
    
    fw=open(out_file,"w")
    for idx in tqdm(range(anom_index,len(scores)), desc='Computing AP...'):
    #for idx in tqdm(range(1,len(scores)), desc='Computing AP...'):
        if(idx%20000) == 0:
            s = scores[:idx+1]
            auc, ap = compute_statistics(s, labels[:idx+1])
            fw.write(str(idx)+"\t"+"{:.4f}".format(ap)+"\t"+"{:.4f}".format(auc)+"\t\n")
            fw.flush()
            
    auc, ap = compute_statistics(scores, labels)
    print "AUC="+str(auc)+" & AP ="+str(ap)
    fw.write(str(idx)+"\t"+"{:.4f}".format(ap)+"\t"+"{:.4f}".format(auc)+"\t\n")
    fw.flush()
    fw.close()
    
    data=np.loadtxt(out_file)
    plt.figure(figsize=(8,6))
    plt.plot(data[:,0],data[:,1],linewidth=3.0)
    plt.xlabel("Num Instances",fontsize=24)
    plt.ylabel("AP",fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=20)
    plt.savefig(out_file+"_AP.pdf")
    plt.close()
    
    return auc,ap

def Analyze_Stream_Data(score_dir,fname,method):
    if method == "HST":
        data = np.loadtxt(os.path.join(score_dir,fname+".csv"),delimiter=",")
        scores = -data[:,1]
        labels = data[:,0].astype(int)
    elif method=="RSH":
        data = np.loadtxt(os.path.join(score_dir,fname+".csv"),delimiter=",")
        scores = -data[:,0]
        labels = data[:,1].astype(int)
    elif method=="LODA":
        data = np.loadtxt(os.path.join(score_dir,fname+".csv"),delimiter=",")
        scores = data[:,0]
        labels = data[:,1].astype(int)
        
    nelements = len(scores)
    print np.sum(labels)
    anom_index = np.where(labels==1.0)[0][0]
    print "Anom Index="+str(anom_index)+" for "+str(fname)
    
    fw=open(os.path.join(score_dir,fname+"_Stats.txt"),"w")
    for idx in tqdm(range(anom_index,len(scores)), desc='Computing AP...'):
    #for idx in tqdm(range(1,len(scores)), desc='Computing AP...'):
        if(idx%20000) == 0:
            s = scores[:idx+1]
            auc, ap = compute_statistics(s, labels[:idx+1])
            fw.write(str(idx)+"\t"+"{:.4f}".format(ap)+"\t"+"{:.4f}".format(auc)+"\t\n")
            fw.flush()
            
    auc, ap = compute_statistics(scores, labels)
    print "AUC="+str(auc)+" & AP ="+str(ap)
    fw.write(str(idx)+"\t"+"{:.4f}".format(ap)+"\t"+"{:.4f}".format(auc)+"\t\n")
    fw.flush()
    fw.close()
    
    data=np.loadtxt(os.path.join(score_dir,fname+"_Stats.txt"))
    plt.figure(figsize=(8,6))
    plt.plot(data[:,0],data[:,1],linewidth=3.0)
    plt.xlabel("Num Instances",fontsize=24)
    plt.ylabel("AP",fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(score_dir,fname+"_AP.pdf"))
    plt.close()
    
    return auc,ap
    
def Analyze_Stream_LODA(score_dir,fname):
    data = np.loadtxt(os.path.join(score_dir,fname+".csv"),delimiter=",")
    scores = data[:,0]
    labels = data[:,1].astype(int)
    nelements = len(scores)
    initial_sample_size = 181877 #int(1.0/100 * nelements)
    f=open(os.path.join(score_dir,fname+"_APScores.txt"),"w")
    #f=open(os.path.join(score_dir,fname+"_AUCScores.txt"),"w")
    for idx in tqdm(range(initial_sample_size, len(scores)), desc="Streaming..."):
        if (idx % 20000) == 0:
            s = scores[:idx+1]
            average_precision = average_precision_score(labels[:idx+1], s)
            #auc = roc_auc_score(labels, scores)
            f.write(str(idx) + "\t" + "{:.4f}".format(average_precision) + "\t\n")
            #f.write(str(idx) + "\t" + "{:.4f}".format(auc) + "\t\n")
            f.flush()
    average_precision = average_precision_score(labels, scores)
    f.write(str(idx) + "\t" + "{:.4f}".format(average_precision) + "\t\n")
    #f.write(str(idx) + "\t" + "{:.4f}".format(auc) + "\t\n")
    f.flush()
    
    data=np.loadtxt(os.path.join(score_dir,fname+"_APScores.txt"))
    #data=np.loadtxt(os.path.join(score_dir,fname+"_AUCScores.txt"))
    plt.figure(figsize=(8,6))
    plt.plot(data[:,0],data[:,1],linewidth=3.0)
    plt.xlabel("Num Instances",fontsize=24)
    plt.ylabel("AP",fontsize=24)
    #plt.ylabel("AUC",fontsize=24)   
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(score_dir,fname+"_AP.pdf"))
    #plt.savefig(os.path.join(score_dir,fname+"_AUC.pdf"))
    
    print "FileName="+str(fname)
    auc,ap = compute_statistics(scores, labels)
    print "AUC="+str(auc)+" & AP="+str(ap)
    return auc,ap
    
def Create_Streaming_Table(base_dir,out_file):
    fw=open(out_file,'w')
    winSize_HttpSmtp = [7206,36034,72069,180173]
    winSize_SpamSMS = [55,278,557,1393]
    
    # Doing for HttpSmtp First
    hstree_dir =  os.path.join(base_dir,"HttpSmtpContinuous","HSTrees4")
    rshash_dir = os.path.join(base_dir,"HttpSmtpContinuous", "RSHash4")
    loda_dir = os.path.join(base_dir,"HttpSmtpContinuous", "LODA4")
    
    fw.write("HttpSmtpContinuous\n")
    perc2={1:7206, 2:36034, 3:72069, 4:180173}
    for perc in [1,5,10,25]:
        hst_auc_arr=[]
        hst_ap_arr=[]
        rsh_auc_arr=[]
        rsh_ap_arr=[]
        loda_auc_arr=[]
        loda_ap_arr=[]
        for run_id in range(1,4):
            hstree_file = os.path.join(hstree_dir,"HST_Run"+str(run_id)+"_"+str(perc)+"Per_AnormalyScore_HttpSmtpContinuous_0.csv")
            data = np.loadtxt(hstree_file,delimiter=",")
            scores = -data[:,1]
            labels = data[:,0].astype(int)
            auc, ap = compute_statistics(scores, labels)
            hst_auc_arr.append(auc)
            hst_ap_arr.append(ap)
            
            
            rshash_file = os.path.join(rshash_dir,"RSH_"+str(perc2[perc])+"_"+str(run_id)+".csv")
            data = np.loadtxt(rshash_file,delimiter=",")
            scores = -data[:,0]
            labels = data[:,1].astype(int)
            auc, ap = compute_statistics(scores, labels)
            rsh_auc_arr.append(auc)
            rsh_ap_arr.append(ap)
            
            loda_file = os.path.join(loda_dir,"LODA_"+str(perc2[perc])+"_"+str(run_id)+".csv")
            data = np.loadtxt(loda_file,delimiter=",")
            scores = data[:,0]
            labels = data[:,1].astype(int)
            auc, ap = compute_statistics(scores, labels)
            loda_auc_arr.append(auc)
            loda_ap_arr.append(ap)
            
        hst_auc_arr =  np.array(hst_auc_arr)
        hst_ap_arr = np.array(hst_ap_arr)
        rsh_auc_arr =  np.array(rsh_auc_arr)
        rsh_ap_arr = np.array(rsht_ap_arr)
        loda_auc_arr =  np.array(loda_auc_arr)
        loda_ap_arr = np.array(loda_ap_arr)
        fw.write(str(perc)+"\t"+str(hst_auc_arr)+","+str(hst_ap_arr))
        fw.write("\t"+str(rsh_auc_arr)+","+str(rsh_ap_arr))
        fw.write("\t"+str(loda_auc_arr)+","+str(loda_ap_arr)+"\n")
        
    
    #Doing for SpamSMS First
    hstree_dir =  os.path.join(base_dir,"SpamSMSCounts","HSTrees3")
    rshash_dir = os.path.join(base_dir,"SpamSMSCounts", "RSHash3")
    loda_dir = os.path.join(base_dir,"SpamSMSCounts", "LODA3")
    fw.write("SpamSMS\n")
    perc2={1:55, 2:278, 3:557, 4:1393}
    for perc in [1,5,10,25]:
        hst_auc_arr=[]
        hst_ap_arr=[]
        rsh_auc_arr=[]
        rsh_ap_arr=[]
        loda_auc_arr=[]
        loda_ap_arr=[]
        for run_id in range(1,4):
            hstree_file = os.path.join(hstree_dir,"HST_Run"+str(run_id)+"_"+str(perc)+"Per_AnormalyScore_HttpSmtpContinuous_0.csv")
            data = np.loadtxt(hstree_file,delimiter=",")
            scores = -data[:,1]
            labels = data[:,0].astype(int)
            auc, ap = compute_statistics(scores, labels)
            hst_auc_arr.append(auc)
            hst_ap_arr.append(ap)
            
            
            rshash_file = os.path.join(rshash_dir,"RSH_"+str(perc2[perc])+"_"+str(run_id)+".csv")
            data = np.loadtxt(rshash_file,delimiter=",")
            scores = -data[:,0]
            labels = data[:,1].astype(int)
            auc, ap = compute_statistics(scores, labels)
            rsh_auc_arr.append(auc)
            rsh_ap_arr.append(ap)
            
            loda_file = os.path.join(loda_dir,"LODA_"+str(perc2[perc])+"_"+str(run_id)+".csv")
            data = np.loadtxt(loda_file,delimiter=",")
            scores = data[:,0]
            labels = data[:,1].astype(int)
            auc, ap = compute_statistics(scores, labels)
            loda_auc_arr.append(auc)
            loda_ap_arr.append(ap)
            
        hst_auc_arr =  np.array(hst_auc_arr)
        hst_ap_arr = np.array(hst_ap_arr)
        rsh_auc_arr =  np.array(rsh_auc_arr)
        rsh_ap_arr = np.array(rsht_ap_arr)
        loda_auc_arr =  np.array(loda_auc_arr)
        loda_ap_arr = np.array(loda_ap_arr)
        fw.write(str(perc)+"\t"+str(hst_auc_arr)+","+str(hst_ap_arr))
        fw.write("\t"+str(rsh_auc_arr)+","+str(rsh_ap_arr))
        fw.write("\t"+str(loda_auc_arr)+","+str(loda_ap_arr)+"\n")
        
    fw.close()
    
    

if __name__ == '__main__':
    #score_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/HTTP/AnormalyScore_http_0.csv"
    #Analyze_Stream_HSTree(score_file)
    in_file = "../../../Streaming_HighDim_Case/Data/parametersNdata_smtp_http.txt"
    out_file = "../../../Streaming_HighDim_Case/Data/smtp_http.csv"
    #convert_from_HSTreeFile(in_file, out_file)
    
    in_file = "../../../Streaming_HighDim_Case/Data/parametersNdata_SpamSmsCounts.txt"
    out_file = "../../../Streaming_HighDim_Case/Data/SpamSmsCounts.csv"
    #convert_from_HSTreeFile(in_file, out_file)
    #print STOP
    
    base_dir = "../../../Streaming_HighDim_Case/"
    out_file = "../../../Streaming_HighDim_Case/"
    Create_Streaming_Table(base_dir,out_file)
    
    #score_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/HttpSmtpContinuous/RSHash3/"
    #Analyze_Stream_Dir(score_dir,"RSH")
    #score_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/S  treaming_HighDim_Case/SpamSMSCounts/LODA2/"
    #Analyze_Stream_Dir(score_dir,"LODA")
    #score_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/SpamSMSCounts/RSHash2/"
    #Analyze_Stream_Dir(score_dir,"RSH")
    #score_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/HttpSmtpContinuous_Shuffled/RSHash/"
    #Analyze_Stream_Dir(score_dir,"RSH")
    #score_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/SpamURL/RSHash/"
    #Analyze_Stream_Dir(score_dir,"RSH")

    #scores = np.loadtxt("/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Plotting_Notebooks/Data/Streaming/SpamURL/LODA/Overall_Sccores.csv",delimiter=",")
    #labels = np.loadtxt("/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Plotting_Notebooks/Data/Streaming/SpamURL/LODA/Overall_Labels.csv",delimiter=",")
    #out_file = "../../../Plotting_Notebooks/Data/Streaming/SpamURL/LODA/DayByDay_SpamURL_Stats.txt"
    #Analyze_Stream_SpamURL(scores, labels,out_file)
    
    '''
    score_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Streaming_HighDim_Case/HttpSmtpContinuous/LODA"
    fname = "LODA_10per_dense_TwoWin_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_10per_dense_Continuous_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_10per_sparse_TwoWin_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_10per_sparse_Continuous_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_256_sparse_TwoWin_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_256_sparse_Continuous_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_1000_sparse_Continuous_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_1000_sparse_Continuous_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_Shuffled_1000_sparse_Continuous_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_Shuffled_1000_sparse_TwoWin_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_Shuffled_256_sparse_Continuous_500"
    Analyze_Stream_LODA(score_dir, fname)
    fname = "LODA_Shuffled_256_sparse_TwoWin_500"
    Analyze_Stream_LODA(score_dir, fname)
    '''