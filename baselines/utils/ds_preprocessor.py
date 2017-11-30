import sys

def preprocess_abalone(in_file, out_file):
    f=open(in_file,"r")
    fw=open(out_file, "w")
    line=f.readline()
    fw.write("G1,G2,G3,V1,V2,V3,V4,V5,V6,V7,Response\n")
    while line:
        line = line.replace("\n","")
        split = line.split(",")
        if(split[0]=="M"):
            fw.write("1,0,0")
        elif(split[0]=="F"):
            fw.write("0,1,0")
        elif(split[0]=="I"):
            fw.write("0,0,1")
        for i in range(1,len(split)):
            fw.write(","+str(split[i]))
        fw.write("\n")
        line = f.readline()
    f.close()
    fw.close()

def preprocess_ecoli(in_file, out_file):
    f=open(in_file, "r")
    fw=open(out_file, "w")
    fw.write("MCG,GVH,LIP,CHG,AAC,ALM1,ALM2,Class\n")
    line=f.readline()
    while line:
        line=line.replace("\n","")
        line=line.replace("   "," ")
        line=line.replace("  "," ")
        split=line.split()
        for j in range(1,len(split)-1):
            fw.write(split[j]+",")
        fw.write(split[len(split)-1]+"\n")
        line=f.readline()
    f.close()
    fw.close()

def preprocess_isolet(in_file1,in_file2,out_file):
    fw=open(out_file,"w")
    f=open(in_file1,"r")
    line=f.readline()
    length=len(line.split(","))
    for i in range(length-1):
        fw.write("V_"+str(i)+",")
    fw.write("Class\n")
    while line:
        fw.write(line)
        line=f.readline()
    f.close()
    f=open(in_file2,"r")
    line=f.readline()
    while line:
        fw.write(line)
        line=f.readline()
    f.close()
    fw.close()

def preprocess_parkinsons(in_file, out_file):
    f=open(in_file,"r")
    fw=open(out_file, "w")
    
    line=f.readline()
    line=line.replace("\n","")
    line=line.replace("\r","")
    split=line.split(",")
    status_column = -1
    
    for i in range(1,len(split)):
        if(split[i]=="status"):
            status_column = i
        else:
            fw.write(split[i]+",")
    fw.write(split[status_column]+"\n")
    line=f.readline()
    
    while line:
        line=line.replace("\n","")
        line=line.replace("\r","")
        split = line.split(",")
        for i in range(1, len(split)):
            if(i!=status_column):
                fw.write(split[i]+",")
        fw.write(split[status_column]+"\n")
        line=f.readline()
    f.close()
    fw.close()
    
in_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/parkinsons/parkinsons.data"
out_file = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/parkinsons/parkinsons_preprocessed"
preprocess_parkinsons(in_file, out_file)

in_file="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/abalone/abalone.txt"
out_file="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/abalone/abalone_HL_prep.txt"
#preprocess_abalone(in_file, out_file)

in_file="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/ecoli/ecoli.data"
out_file="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/ecoli/ecoli_preprocessed.txt"
#preprocess_ecoli(in_file, out_file)

in_file1="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/isolet/isolet1+2+3+4.data"
in_file2="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/isolet/isolet5.data"
out_file="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/Benchmark_Datasets/isolet/isolet.data"
#preprocess_isolet(in_file1,in_file2,out_file)