import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.style.use(hep.style.ROOT)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

import argparse

parser = argparse.ArgumentParser(description='Plotting MoE Stacked Bars by PID')
parser.add_argument('-m', '--model', type=str, required=True, help='Model name, (10_pct, seed_0, seed_1)')
parser.add_argument('--pid', action='store_true', help='Whether to plot by particle type')

model_name = parser.parse_args().model
pid_plotting = parser.parse_args().pid

import subprocess
import os

num_experts = model_name.split('_k')[0].split('n')[1]
num_experts = int(num_experts)
expert_weight_data = [np.array([]) for _ in range(num_experts)]
agglomerated_data = [np.array([]) for _ in range(num_experts)]

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

part_type = ['Charged_Hadron', 'Neutral_Hadron', 'Photon', 'Electron', 'Muon']

model_name_split = model_name.split('.')[0]
data_dir = f'/moe-interpretability-pv/stacked_bars_pid_{model_name_split}/'

for part_idx, part_type in enumerate(part_type):
    for file in os.listdir(data_dir):
        if file.endswith('.npy') and 'assignments' not in file:
            if part_type.lower() in file:
                expert_idx_in_file = int(file.split('expert_')[1].split('_')[0])
                data = np.load(data_dir+file, allow_pickle=True)
                for expert_idx in range(num_experts):
                    expert_weight_data[expert_idx_in_file] = np.concatenate((expert_weight_data[expert_idx_in_file], data))
        else:
            continue
    
    print('data loaded for ', part_type)
    num_bins = 200
    bottom = np.zeros(num_bins)
    
    agglomerated_data = [np.append(agglomerated_data[expert_idx], expert_weight_data[expert_idx]) for expert_idx in range(num_experts)]
    total_counts = sum([len(expert_weight_data[expert_idx]) for expert_idx in range(num_experts)])
    proportions = np.array([len(expert_weight_data[expert_idx]) / total_counts for expert_idx in range(num_experts)])
    percentages = np.round(proportions*100, 1)
    #remove experts with 0 assignments from proportions
    proportions = proportions[proportions > 0]
    expert_entropy = np.round(-np.sum(np.log(proportions)*proportions),3)
    print(f'Expert weight distribution for {part_type} jets has entropy: {expert_entropy}')
    fig, ax = plt.subplots()
    ax.hist(expert_weight_data, bins=num_bins, histtype='barstacked', label=[f'Expert {i}: {percentages[i]}%' for i in range(num_experts)], density=True, color=plt.cm.tab10.colors[:num_experts])
    ax.legend(fontsize=18)
    ax.set_title(f'Expert Weights, {part_type.replace("_", " ")} Particles \n Entropy: {np.round(expert_entropy, 3)}')
    plt.savefig(f'./MoeStackedBar_plot_{part_type}_100k.png')
    plt.close()

    fig, ax = plt.subplots()
    expert_blind = np.array([])
    # get all expert weights into one array
    for expert_idx in range(num_experts):
        expert_blind = np.concatenate((expert_blind, expert_weight_data[expert_idx]))
    ax.hist(expert_blind, bins=num_bins, histtype='bar', color='tab:blue', alpha=1, label='All Experts', density=True)
    ax.legend(fontsize=18, loc='best')
    ax.set_title(f'Expert Weights, {part_type.replace("_", " ")} Particles')
    plt.savefig(f'./MoeAgglomeratedBar_plot_{part_type}_100k.png')
    plt.close()

    fig, ax = plt.subplots()
    for expert_idx in range(num_experts):
        # only plot experts >= 1%
        if 1 <= percentages[expert_idx]:
            ax.hist(expert_weight_data[expert_idx], bins=num_bins//6, histtype='step', label=f'Expert {expert_idx}', density=True, color=plt.cm.tab10.colors[expert_idx])
    ax.legend(fontsize=18)
    ax.set_title(f'Separately Normalized Expert Weights, {part_type.replace("_", " ")}')
    plt.savefig(f'./SeparateExperts_plot_{part_type}_100k.png')
    plt.close()
    
    expert_weight_data = [np.array([]) for _ in range(num_experts)]
    # subprocess.run(['sudo', 'mv', f'./MoeStackedBar_plot_{label}_100k.png', f'/moe-interpretability-pv/moe_stacked_bars_100k_data/MoeStackedBar_plot_{label}_100k.png'])
    # data_0_to_1000_jet_type_Higgs BB_expert_0_stacked_MoE_bars_100k.npy

total_counts = sum([len(agglomerated_data[expert_idx]) for expert_idx in range(num_experts)])
percentages = np.round(np.array([len(agglomerated_data[expert_idx]) / total_counts * 100 for expert_idx in range(num_experts)]), 1)
expert_entropy = -np.sum(np.log(percentages/100)*percentages/100)

fig, ax = plt.subplots()
ax.hist(agglomerated_data, bins=num_bins, histtype='barstacked', label=[f'Expert {i}: {percentages[i]}%' for i in range(num_experts)], density=True, color=plt.cm.tab10.colors[:num_experts])
ax.legend(fontsize=18, loc='best')
ax.set_title(f'Expert Weights, All Particles')
plt.savefig(f'./MoeStackedBar_plot_all_100k.png')
plt.close()

fig, ax = plt.subplots()
# get all expert weights into one array
agglomerated_expert_blind = np.array([])
for expert_idx in range(num_experts):
    agglomerated_expert_blind = np.concatenate((agglomerated_expert_blind, agglomerated_data[expert_idx]))
ax.hist(agglomerated_expert_blind, bins=num_bins, histtype='bar', color='tab:blue', alpha=0.6, label='All Experts', density=True)
ax.legend(fontsize=18)
ax.set_title(f'Expert Weights, All Particles')
plt.savefig(f'./MoeAgglomeratedBar_plot_all_100k.png')
plt.close()

for expert_idx in range(num_experts):
    fig, ax = plt.subplots()
    ax.hist(agglomerated_data[expert_idx], bins=num_bins, range=(0,1), histtype='step', label=f'Expert {expert_idx}', density=True, color=plt.cm.tab10.colors[expert_idx])
    ax.legend(fontsize=18)
    ax.set_title(f'Expert {expert_idx} Weights, All Particles')
    plt.savefig(f'./SeparateExperts_plot_expert_{expert_idx}_100k.png')
    plt.close()