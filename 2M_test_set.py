import os
import subprocess
import numpy as np

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
    output_path = f'/moe-interpretability-pv/datasets/jc_full_2M_{attribute}.npy'
    np.save(output_path, full_array)
    print(f'Saved combined array for {attribute} to {output_path}, Shape: {full_array.shape}, Dtype: {full_array.dtype}')
