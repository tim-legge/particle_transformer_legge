import os
import subprocess
import numpy as np

data_dir = '/moe-interpretability-pv/datasets/'

while 10>11:
    for attribute in ['vectors', 'points', 'features', 'mask', 'labels']:
        full_array = None
        file_list = sorted([file for file in os.listdir('/moe-interpretability-pv/datasets/2M_test/') if file.endswith(f'_{attribute}.npy')])
        print(f'Processing attribute: {attribute}, number of files: {len(file_list)}')
        for file in file_list:
            file_path = os.path.join('/moe-interpretability-pv/datasets/2M_test/', file)
            data = np.load(file_path, allow_pickle=True)
            print(f'File: {file}, Shape: {data.shape}, Dtype: {data.dtype}')
            if full_array is None:
                full_array = data
            else:
                full_array = np.concatenate((full_array, data), axis=0)
        output_file = f'jc_full_2M_{attribute}.npy'
        np.save(output_file, full_array)
        subprocess.run(['sudo', 'mv', output_file, os.path.join(data_dir, output_file)])
        print(f'Saved combined array for {attribute} to {os.path.join(data_dir, output_file)}, Shape: {full_array.shape}, Dtype: {full_array.dtype}')
    
feats_0 = np.load('/moe-interpretability-pv/datasets/jc_full_2M_features_0.npy', allow_pickle=True)
feats_1 = np.load('/moe-interpretability-pv/datasets/jc_full_2M_features_1.npy', allow_pickle=True)
features = np.concatenate((feats_0, feats_1), axis=0)
np.save('jc_full_2M_features.npy', features)
subprocess.run(['sudo', 'mv', 'jc_full_2M_features.npy', os.path.join(data_dir, 'jc_full_2M_features.npy')])