import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.style.use(hep.style.ROOT)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


import argparse

parser = argparse.ArgumentParser(description='Expert Ablation')
parser.add_argument('-m', '--model', type=str, required=True, help='Model name, (100k, seed_0, seed_1)')

model = parser.parse_args().model

import subprocess
import os

num_experts = 8
expert_weight_data = [np.array([]) for _ in range(num_experts)]
agglomerated_data = expert_weight_data.copy()

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

data_dir = f'/moe-interpretability-pv/moe_stacked_bars_100k_data_{model}_ll/'

for label_idx, label in idx_to_label.items():
    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
                expert_idx_in_file = int(file.split('expert_')[1].split('_')[0])
                data = np.load(data_dir+f'{file}', allow_pickle=True)
                for expert_idx in range(num_experts):
                    expert_weight_data[expert_idx_in_file] = np.concatenate((expert_weight_data[expert_idx_in_file], data))
        else:
            continue
    
    print('data loaded for ', label)
    num_bins = 200
    bottom = np.zeros(num_bins)
    
    agglomerated_data = [np.append(agglomerated_data[expert_idx], expert_weight_data[expert_idx]) for expert_idx in range(num_experts)]
    total_counts = sum([len(expert_weight_data[expert_idx]) for expert_idx in range(num_experts)])
    percentages = np.round(np.array([len(expert_weight_data[expert_idx]) / total_counts * 100 for expert_idx in range(num_experts)]), 1)

    fig, ax = plt.subplots()
    ax.hist(expert_weight_data, bins=num_bins, histtype='barstacked', label=[f'Expert {i}: {percentages[i]}%' for i in range(num_experts)], density=True, color=plt.cm.tab10.colors[:num_experts])
    ax.legend(fontsize=18, loc='upper left')
    ax.set_title(f'Distribution of Expert Weights, {label.replace("_", " ")} Jets')
    plt.savefig(f'./MoeStackedBar_plot_{label}_100k.png')
    plt.close()

    fig, ax = plt.subplots()
    expert_blind = np.array([])
    # get all expert weights into one array
    for expert_idx in range(num_experts):
        expert_blind = np.concatenate((expert_blind, expert_weight_data[expert_idx]))
    ax.hist(expert_blind, bins=num_bins, histtype='bar', color='tab:blue', alpha=1, label='All Experts', density=True)
    ax.legend(fontsize=18, loc='upper left')
    ax.set_title(f'Distribution of Expert Weights, {label.replace("_", " ")} Jets')
    plt.savefig(f'./MoeAgglomeratedBar_plot_{label}_100k.png')
    plt.close()

    fig, ax = plt.subplots()
    for expert_idx in range(num_experts):
        # only plot 4 most significant experts
        if sorted(percentages)[-4] <= percentages[expert_idx]:
            ax.hist(expert_weight_data[expert_idx], bins=num_bins//6, histtype='step', label=f'Expert {expert_idx}', density=True, color=plt.cm.tab10.colors[expert_idx])
    ax.legend(fontsize=18, loc='upper left')
    ax.set_title(f'Separately Normalized Expert Weights, {label.replace("_", " ")}')
    plt.savefig(f'./SeparateExperts_plot_{label}_100k.png')
    plt.close()
    
    expert_weight_data = [np.array([]) for _ in range(num_experts)]
    # subprocess.run(['sudo', 'mv', f'./MoeStackedBar_plot_{label}_100k.png', f'/moe-interpretability-pv/moe_stacked_bars_100k_data/MoeStackedBar_plot_{label}_100k.png'])
    # data_0_to_1000_jet_type_Higgs BB_expert_0_stacked_MoE_bars_100k.npy

total_counts = sum([len(agglomerated_data[expert_idx]) for expert_idx in range(num_experts)])
percentages = np.round(np.array([len(agglomerated_data[expert_idx]) / total_counts * 100 for expert_idx in range(num_experts)]), 1)

fig, ax = plt.subplots()
ax.hist(agglomerated_data, bins=num_bins, histtype='barstacked', label=[f'Expert {i}: {percentages[i]}%' for i in range(num_experts)], density=True, color=plt.cm.tab10.colors[:num_experts])
ax.legend(fontsize=18, loc='upper left')
ax.set_title(f'Distribution of Expert Weights, All Jets')
plt.savefig(f'./MoeStackedBar_plot_all_100k.png')
plt.close()

fig, ax = plt.subplots()
# get all expert weights into one array
agglomerated_expert_blind = np.array([])
for expert_idx in range(num_experts):
    agglomerated_expert_blind = np.concatenate((agglomerated_expert_blind, agglomerated_data[expert_idx]))
ax.hist(agglomerated_expert_blind, bins=num_bins, histtype='bar', color='tab:blue', alpha=0.6, label='All Experts', density=True)
ax.legend(fontsize=18, loc='upper left')
ax.set_title(f'Distribution of Expert Weights, All Jets')
plt.savefig(f'./MoeAgglomeratedBar_plot_all_100k.png')
plt.close()

for expert_idx in range(num_experts):
    fig, ax = plt.subplots()
    ax.hist(agglomerated_data[expert_idx], bins=num_bins, range=(0,1), histtype='step', label=f'Expert {expert_idx}', density=True, color=plt.cm.tab10.colors[expert_idx])
    ax.legend(fontsize=18, loc='upper left')
    ax.set_title(f'Expert {expert_idx} Weights, All Jets')
    plt.savefig(f'./SeparateExperts_plot_expert_{expert_idx}_100k.png')
    plt.close()