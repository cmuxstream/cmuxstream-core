import os
import sys
import numpy as np
import pandas as pd

def consolidate_datasets(in_dir,out_dir):
    ds_files = os.listdir(in_dir)
    for ds_file in ds_files:
        ds_dir = os.path.join(in_dir, ds_file)
        anom_files=['very_easy','easy','medium','hard','very_hard']
        normal_file = os.path.join(ds_dir,"normal.txt")
        X_normal = pd.read_csv(normal_file,delimiter=' ',header=False)
        print X_normal.shape
        

in_dir = "/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/LODA_Datasets/numerical
consolidate_datasets()
        
            