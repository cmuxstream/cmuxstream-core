import numpy as np
import os,sys
import pandas as pd
from docutils.nodes import important
from os.path import basename

def read_dataset(in_file):
    data = np.loadtxt(in_file, delimiter=',')
    X = data[:,0:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    return X, y

def noisy_signal(data, desired_SNR_dB):
    '''
    Input: A dataframe and desired snr level
    Output: Noisy signal with desired snr level
    '''
    data = data.values
    variance = np.var(data, axis=0)
    cov = np.diag(variance)
    #mean = np.zeros(len(variance))
    mean = np.mean(data, axis=0)
    

    noise = np.random.multivariate_normal(mean, cov, (data.shape[0],))

    signal_power = np.mean(np.square(data), axis=0)
    noise_power = np.mean(np.square(noise), axis=0)
    # Scale factor
    scale_factor = (signal_power / noise_power) * (10 ** (- desired_SNR_dB / 10))
    # scale noise to have desired SNR when added to the data
    noise = np.nan_to_num(noise * np.sqrt(scale_factor))
    return data + noise

def create_privileged_dataset(df, outliers_fraction, desired_SNR_db, label='label'):
    df_anom = df.loc[df[label] == 1]
    df = df.loc[df[label] == 0]
    

    cols = list(set(list(df.columns.values)) - set([label]))
    num_cols = len(cols)

    # designating <= 20% of features as important features (the fraction of features to perturb can be made a function argument)
    important_features = np.unique(np.random.choice(cols, num_cols // 5))
    # following if is added just to make sure we have atleast one important feature
    if len(important_features) < 1:
    	important_features = np.unique(np.random.choice(cols, 1))
    residual_features = [x for x in cols if x not in important_features]
    print("important features", important_features)

    msk = np.random.rand(len(df)) < 1 - outliers_fraction
    normal = df[msk]
    noisy_normal = df[~msk]
    normal.reset_index()
    noisy_normal.reset_index()
    # add noise only to the points selected as noisy_normal points
    noisy_normal[important_features] = noisy_signal(noisy_normal[important_features].copy(),
                                                    desired_SNR_dB=desired_SNR_db)
    # add class label as 1 for these noisy points
    noisy_normal.loc[:, label] = 1
    # append these noisy points with the normal points and return the dataframe along with perturbed features(important_features) and remaining features
    #prepared_df = pd.concat([normal, noisy_normal, df_anom], ignore_index=True)
    prepared_df = pd.concat([normal, noisy_normal], ignore_index=True)
    
    return prepared_df, important_features, residual_features

def main():
    in_file = sys.argv[1]
    outlier_fraction = float(sys.argv[2])
    SnR = float(sys.argv[3])
    out_dir = sys.argv[4]
    
    X,y = read_dataset(in_file)
    df = pd.DataFrame(X)
    df['Labels']=y
    prepared_df, imp_features, residual_features = create_privileged_dataset(df, outlier_fraction, SnR, 'Labels')
    
    modified_data = prepared_df.as_matrix()
    print X.shape, modified_data.shape
    out_file = os.path.join(out_dir,basename(in_file)+"_"+str(outlier_fraction)+"_"+str(SnR)+"_NOISY")
    np.savetxt(out_file, modified_data, delimiter=',')
    
if __name__ == '__main__':
    main()
