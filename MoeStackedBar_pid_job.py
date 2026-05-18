# Moe ParT modified to store router output, accessed via MoeRouterHook

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

from functools import partial
from weaver.utils.logger import _logger
import os
import uproot
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch._torch_docs import reproducibility_notes, sparse_support_notes, tf32_notes
import mplhep as hep
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.style.use(hep.style.ROOT)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

import subprocess
import logging
import argparse
import model_utils as mu

total_chunks = 10

parser = argparse.ArgumentParser(description='Running inference to study first-layer experts')

parser.add_argument('-m', '--model', type=str, required=True, help='Model name, must be inside MoE_Interpretability/models/')
parser.add_argument('-e', '--num-experts', type=int, required=False, default=8, help='Number of experts in the MoE (default 8)')
parser.add_argument('-k', '--k-experts', type=int, required=False, default=2, help='Number of experts selected by the router (default 4)')
parser.add_argument('-ffn', '--ffn-ratio', type=int, required=False, default=4, help='FFN expansion ratio (default 4)')

parser.add_argument('-c', '--chunk', type=int, required=True, help=f'Which chunk of the dataset to run on (0-{total_chunks-1})')
parser.add_argument('-n', '--num-jets', type=int, required=False, 
                    default=1000, help='number of jets to run inference on per step (default 1000)')
parser.add_argument('-l', '--layer', type=int, required=False, default=0, help='Which MoE layer to study (default 0, first MoE layer)')
parser.add_argument('-r', '--restart', action='store_true', help='Whether to restart the job if previous results exist')

model_name = parser.parse_args().model
num_experts = parser.parse_args().num_experts
k_experts = parser.parse_args().k_experts
ffn_ratio = parser.parse_args().ffn_ratio
chunk = parser.parse_args().chunk
num_jets = parser.parse_args().num_jets
moe_layer = parser.parse_args().layer
restart = parser.parse_args().restart

idx_to_label = {
    0: 'Higgs_BB',
    1: 'Higgs_CC',
    2: 'Higgs_GG',
    3: 'Higgs_QQL',
    4: 'Higgs_4Q',
    5: 'Top_BL',
    6: 'Top_BQQ',
    7: 'W_QQ',
    8: 'QCD',
    9: 'Z_QQ'
}

model_path = os.path.join('models/', model_name)
model_name_split = model_name.split('.')[0]
data_dir = f'/moe-interpretability-pv/stacked_bars_pid_{model_name_split}/'

model = mu.get_moe_model(data_type='jc_full', moe_num_experts=num_experts, moe_top_k=k_experts, ffn_ratio=ffn_ratio, trim=False)[0]

state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)
maxjets = 100000
start_idx = chunk*(maxjets//total_chunks)

counter_file = f'counter_stacked_bars_pid_chunk_{chunk}.txt'

if not os.path.exists(data_dir):
    subprocess.run(['sudo', 'mkdir', '-p', data_dir])

if not os.path.exists(data_dir+counter_file) or restart:
    with open(counter_file, 'w') as f:
        f.write(f'{start_idx}')
    subprocess.run(['sudo', 'mv', '-f', counter_file, data_dir])

with open(data_dir+counter_file, 'r') as f:
    start_idx = int(f.read().strip())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature_ids = {
    0: 'part_pt_log',
    1: 'part_e_log',
    2: 'part_logptrel',
    3: 'part_logerel',
    4: 'part_deltaR',
    5: 'part_charge',
    6: 'part_isChargedHadron',
    7: 'part_isNeutralHadron',
    8: 'part_isPhoton',
    9: 'part_isElectron',
    10: 'part_isMuon',
    11: 'part_d0',
    12: 'part_d0err',
    13: 'part_dz',
    14: 'part_dzerr',
    15: 'part_deta',
    16: 'part_dphi'
}

part_type = ['charged_hadron', 'neutral_hadron', 'photon', 'electron', 'muon']

howmanyjets = num_jets

while start_idx < (chunk + 1)*(maxjets//total_chunks):
    print(f'Chunk {chunk}: Beginning at index {start_idx}/100000...')
    logging.info(f'Chunk {chunk}: Beginning at index {start_idx}/100000...')
    
    features, labels, masks, points, vectors = mu.load_jet_data(start_idx, start_idx+howmanyjets, data_dir=data_dir, feats='full')

    jet_type = idx_to_label[start_idx // 10000]
    
    router_hook = mu.Router_Hook(model=model, layer=moe_layer)
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(points),torch.from_numpy(features),
                                    torch.from_numpy(vectors),torch.from_numpy(masks))
    print('Inference complete!')
    assignments = router_hook.expert_assignments
    weights = router_hook.expert_weights

    # organize features into 2D array (particles, features)        
    flat_features = features.transpose(0,2,1) # (N, C, P) -> (N, P, C)
    # features: (N, P, C) -> (N*P, C)
    flat_features = flat_features.reshape(-1, features.shape[1])
    flat_features = flat_features[router_hook.valid_indices,:]
    # lists of indices where each particle type appears
    particle_indices = []
    # ordering is charged_hadron, neutral_hadron, photon, electron, muon
    particle_indices.append(np.where(flat_features[:, 6] == 1)[0])
    particle_indices.append(np.where(flat_features[:, 7] == 1)[0])
    particle_indices.append(np.where(flat_features[:, 8] == 1)[0])
    particle_indices.append(np.where(flat_features[:, 9] == 1)[0])
    particle_indices.append(np.where(flat_features[:, 10] == 1)[0])

    # for each particle type, save expert weights as stacked bar data
    for part_type_idx, indices in enumerate(particle_indices):
        type = part_type[part_type_idx]
        print(f'Processing particle type: {type}...')
        weights_by_part = router_hook.expert_weights[indices,:]
        stacking_data = [[weight for weight in weights_by_part[:,i].numpy() if weight != 0] for i in range(router_hook.model.moe_num_experts)]
        for expert_idx, weights in enumerate(stacking_data):
            file_name = f'data_{start_idx}_to_{start_idx+howmanyjets}_part_type_{type}_expert_{expert_idx}_stacked_MoE_bars_100k.npy'
            np.save(file_name, np.array(weights))
            # get filesize
            filesize = subprocess.check_output(['du', '-h', file_name]).split()[0].decode('utf-8')
            #print(f'Saved expert {expert_idx} data for jets {start_idx} to {start_idx+1000}. File size: {filesize}')
            subprocess.run(['sudo', 'mv', file_name, data_dir])
        # saving expert assignments
        assignments_by_part = router_hook.expert_assignments[indices,:]
        assignment_file_name = f'data_{start_idx}_to_{start_idx+howmanyjets}_part_type_{type}_expert_assignments_stacked_MoE_bars_100k.npy'
        np.save(assignment_file_name, assignments_by_part.numpy())
        subprocess.run(['sudo', 'mv', assignment_file_name, data_dir])

    print(f'Iteration {start_idx}/100000 complete! Saved by particle types, rerunning for next iteration...')
    
    start_idx += howmanyjets

    with open(counter_file, 'w') as f:
        f.write(str(start_idx))

    subprocess.run(['sudo', 'mv', '-f', counter_file, f'{data_dir}'])