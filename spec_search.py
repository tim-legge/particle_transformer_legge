
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import sys
from sklearn.decomposition import PCA
from typing import List, Optional
import timeit
import awkward as ak
import torch
import torch.nn as nn
from torch.nn import Parameter 
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
import torch
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F
from typing import Optional, Tuple
_is_fastpath_enabled: bool = True
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)
linear = torch._C._nn.linear
import math
import random
import warnings
import copy
from torch._C import _add_docstr, _infer_size
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable

from functools import partial
from weaver.utils.logger import _logger
import os
import uproot
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch._torch_docs import reproducibility_notes, sparse_support_notes, tf32_notes

import subprocess
import logging
import argparse
import model_utils as mu

total_chunks = 1

parser = argparse.ArgumentParser(description='Running inference to study first-layer experts')

parser.add_argument('-m', '--model', type=str, required=True, help='Model name, must be inside MoE_Interpretability/models/')
parser.add_argument('-e', '--num-experts', type=int, required=False, default=8, help='Number of experts in the MoE (default 8)')
parser.add_argument('-k', '--k-experts', type=int, required=False, default=2, help='Number of experts selected by the router (default 4)')
parser.add_argument('-ffn', '--ffn-ratio', type=int, required=False, default=4, help='FFN expansion ratio (default 4)')
parser.add_argument('-cap', '--capacity-factor', type=int, required=False, default=2, help='Capacity factor for the MoE (default 2)')

parser.add_argument('-c', '--chunk', type=int, required=True, help=f'Which chunk of the dataset to run on (0-{total_chunks-1})')
parser.add_argument('-n', '--num-jets', type=int, required=False, 
                    default=1000, help='number of jets to run inference on per step (default 1000)')
parser.add_argument('-l', '--layer', type=int, required=False, default=0, help='Which MoE layer to study (default 0, first MoE layer)')
parser.add_argument('-r', '--restart', action='store_true', help='Whether to restart the job if previous results exist')

parser.add_argument('--dataset-dir', type=str, required=False, default='/moe-interpretability-pv/datasets/', help='Directory to store intermediate data files (default /moe-interpretability-pv/datasets/)')

model_name = parser.parse_args().model
num_experts = parser.parse_args().num_experts
k_experts = parser.parse_args().k_experts
ffn_ratio = parser.parse_args().ffn_ratio
chunk = parser.parse_args().chunk
num_jets = parser.parse_args().num_jets
moe_layer = parser.parse_args().layer
restart = parser.parse_args().restart
dataset_dir = parser.parse_args().dataset_dir
capacity_factor = parser.parse_args().capacity_factor

model = mu.get_moe_model('jc_full', moe_num_experts=num_experts, moe_top_k=k_experts, 
                         ffn_ratio=ffn_ratio, moe_capacity_factor=capacity_factor)[0]
model_path = os.path.join('models/', model_name)
model_name_split = model_name.split('.')[0]
data_dir = f'/moe-interpretability-pv/spec_search_{model_name_split}/'

state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)
maxjets = 100000
start_idx = chunk*(maxjets//total_chunks)

counter_file = f'counter_stacked_bars_pid_chunk_{chunk}.txt'
partitions_shape = (4, 4, 3, 5, num_experts)
cumulative_partitions_file = f'cumulative_partitions.npy'
cumulative_partitions = np.zeros(partitions_shape)
cumulative_perm_stats_file = f'cumulative_perm_stats.npy'
cumulative_perm_stats = np.zeros(tuple(list(partitions_shape[:-1]) + [3]))

if not os.path.exists(data_dir):
    subprocess.run(['sudo', 'mkdir', '-p', data_dir])

if not os.path.exists(data_dir+counter_file) or restart:
    with open(counter_file, 'w') as f:
        f.write(f'{start_idx}')
    subprocess.run(['sudo', 'mv', '-f', counter_file, data_dir])

with open(data_dir+counter_file, 'r') as f:
    start_idx = int(f.read().strip())

if not os.path.exists(data_dir+cumulative_partitions_file) or restart or not os.path.exists(data_dir+cumulative_perm_stats_file):
    np.save(cumulative_partitions_file, cumulative_partitions)
    np.save(cumulative_perm_stats_file, cumulative_perm_stats)
    subprocess.run(['sudo', 'mv', '-f', cumulative_partitions_file, data_dir])
    subprocess.run(['sudo', 'mv', '-f', cumulative_perm_stats_file, data_dir])
else:
    cumulative_partitions = np.load(data_dir+cumulative_partitions_file)
    cumulative_perm_stats = np.load(data_dir+cumulative_perm_stats_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

howmanyjets = num_jets

features, labels, masks, points, vectors = mu.load_jet_data(stop=1000, step=100, data_dir=dataset_dir, feats='full')

router_hook = mu.Router_Hook(model)
model.eval()
with torch.no_grad():
    _ = model(torch.from_numpy(points),torch.from_numpy(features),
                                torch.from_numpy(vectors),torch.from_numpy(masks))

flat_features = features.transpose(0,2,1) # (N, C, P) -> (N, P, C)
# features: (N, P, C) -> (N*P, C)
flat_features = flat_features.reshape(-1, features.shape[1])
flat_features = flat_features[router_hook.valid_indices,:]

pt = flat_features[:,0]
delta_R = flat_features[:,4]

pt_q1 = np.percentile(pt, 25)
pt_q2 = np.percentile(pt, 50)
pt_q3 = np.percentile(pt, 75)

delta_R_q1 = np.percentile(delta_R, 25)
delta_R_q2 = np.percentile(delta_R, 50)
delta_R_q3 = np.percentile(delta_R, 75)

while start_idx < (chunk + 1)*(maxjets//total_chunks):
    print(f'Chunk {chunk}: Beginning at index {start_idx}/100000...')
    logging.info(f'Chunk {chunk}: Beginning at index {start_idx}/100000...')
    
    features, labels, masks, points, vectors = mu.load_jet_data(start_idx, start_idx+howmanyjets, data_dir=dataset_dir, feats='full')

    router_hook = mu.Router_Hook(model)
    model.eval()
    with torch.no_grad():
        _ = model(torch.from_numpy(points),torch.from_numpy(features),
                                    torch.from_numpy(vectors),torch.from_numpy(masks))

    flat_features = features.transpose(0,2,1) # (N, C, P) -> (N, P, C)
    # features: (N, P, C) -> (N*P, C)
    flat_features = flat_features.reshape(-1, features.shape[1])
    flat_features = flat_features[router_hook.valid_indices,:]

    pt = flat_features[:,0]
    energy = flat_features[:,1]
    pt_rel = flat_features[:,2]
    e_rel = flat_features[:,3]
    delta_R = flat_features[:,4]
    charge = flat_features[:,5]
    pid = flat_features[:,6:11]
    pid = np.argmax(pid, axis=1) # 0: charged_hadron, 1: neutral_hadron, 2: photon, 3: electron, 4: muon

    particle_q_ids = np.zeros((len(pt), 4))
    for particle in range(len(pt)):
        # determine which quartiles each particle belongs to
        pt_quartile = 1 if pt[particle] < pt_q1 else (2 if pt[particle] < pt_q2 else (3 if pt[particle] < pt_q3 else 4))
        delta_R_quartile = 1 if delta_R[particle] < delta_R_q1 else (2 if delta_R[particle] < delta_R_q2 else (3 if delta_R[particle] < delta_R_q3 else 4))
        charge_val = charge[particle]
        pid_val = pid[particle]
        #    particle_q_ids[particle] = [pt_quartile, energy_quartile, pt_rel_quartile, e_rel_quartile, delta_R_quartile, charge_id]
        particle_q_ids[particle] = [pt_quartile, delta_R_quartile, charge_val, pid_val]

    weights = router_hook.expert_weights
    assignments = router_hook.expert_assignments

    id_ranges = np.zeros(particle_q_ids.shape[-1], dtype=object)
    for feature in range(particle_q_ids.shape[-1]):
        # make the feature values start from 0 to function as indices
        id_min = int(particle_q_ids[:, feature].min())
        id_max = int(particle_q_ids[:, feature].max())
        particle_q_ids[:, feature] -= id_min
        id_ranges[feature] = id_max - id_min + 1

    id_ranges = id_ranges.tolist()
    id_ranges.append(num_experts)
    id_ranges = tuple(id_ranges)
    particle_partition = np.zeros(id_ranges)
    particle_partition
    
    stats_shape = tuple(list(id_ranges[:-1]) + [3])
    perm_stats = np.zeros((stats_shape))
    for perm, _ in np.ndenumerate(particle_partition[:-1]):

        perm_slice = np.where(np.all(particle_q_ids == perm[:-1], axis=1))[0]
        perm_weights = weights[perm_slice].numpy()
        perm_assignments = assignments[perm_slice].numpy()
        assignment_dist = np.sum(perm_assignments, axis=0) / np.sum(perm_assignments)
        particle_partition[perm[:-1]] = assignment_dist
        dist_entropy = -np.sum(assignment_dist * np.log(assignment_dist + 1e-10))
        max_expert = np.max(assignment_dist)
        perm_stats[perm[:-1]] = [dist_entropy, max_expert, len(perm_weights)]

        #update cumulative partitions and stats
        if np.sum(perm_assignments) > 0:
            if np.sum(cumulative_perm_stats[perm[:-1],2]) == 0:
                cumulative_partitions[perm[:-1]] = assignment_dist
                cumulative_perm_stats[perm[:-1],0] = np.max(cumulative_partitions[perm[:-1]])
                cumulative_perm_stats[perm[:-1],1] = -np.sum(cumulative_partitions[perm[:-1]] * np.log(cumulative_partitions[perm[:-1]] + 1e-10))
                cumulative_perm_stats[perm[:-1],2] = len(perm_weights)
            else:
                cumulative_partitions[perm[:-1]] = (cumulative_partitions[perm[:-1]]*cumulative_perm_stats[perm[:-1],2] + assignment_dist*len(perm_weights))/(cumulative_perm_stats[perm[:-1],2] + len(perm_weights))
                cumulative_perm_stats[perm[:-1],0] = np.max(cumulative_partitions[perm[:-1]])
                cumulative_perm_stats[perm[:-1],1] = -np.sum(cumulative_partitions[perm[:-1]] * np.log(cumulative_partitions[perm[:-1]] + 1e-10))
                cumulative_perm_stats[perm[:-1],2] += len(perm_weights)
    
    np.save(cumulative_partitions_file, cumulative_partitions)
    np.save(cumulative_perm_stats_file, cumulative_perm_stats)
    subprocess.run(['sudo', 'mv', '-f', cumulative_partitions_file, data_dir])
    subprocess.run(['sudo', 'mv', '-f', cumulative_perm_stats_file, data_dir])
    start_idx += howmanyjets
    with open(counter_file, 'w') as f:
        f.write(f'{start_idx}')
    subprocess.run(['sudo', 'mv', '-f', counter_file, data_dir])

    