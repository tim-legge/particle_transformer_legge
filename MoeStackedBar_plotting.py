import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.style.use(hep.style.ROOT)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

import subprocess
import os

num_experts = 8
expert_weight_data = [np.array([]) for _ in range(num_experts)]

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

for label_idx, label in idx_to_label.items():
    for file in os.listdir('/moe-interpretability-pv/moe_stacked_bars_100k_data'):
        if f'jet_type_{label}' in file:
            if 'attempt_1' in file:
                expert_idx_in_file = int(file.split('expert_')[1].split('_')[0])
                data = np.load(f'/moe-interpretability-pv/moe_stacked_bars_100k_data/{file}', allow_pickle=True)
                for expert_idx in range(num_experts):
                    expert_weight_data[expert_idx_in_file] = np.concatenate((expert_weight_data[expert_idx_in_file], data))
        else:
            continue
    
    print('data loaded for ', label)
    num_bins = 300
    bottom = np.zeros(num_bins)
    
    total_counts = sum([len(expert_weight_data[expert_idx]) for expert_idx in range(num_experts)])
    percentages = [len(expert_weight_data[expert_idx]) / total_counts * 100 for expert_idx in range(num_experts)]

    fig, ax = plt.subplots()
    ax.hist(expert_weight_data, bins=num_bins, histtype='barstacked', label=[f'Expert {i}: {percentages[i]}' for i in range(num_experts)], density=True, color=plt.cm.tab10.colors[:num_experts])
    ax.legend(fontsize=18, loc='upper left')
    ax.set_title(f'Distribution of Expert Weights, {label.replace("_", " ")} Jets')

    plt.savefig(f'./MoeStackedBar_plot_{label}_100k.png')
    expert_weight_data = [np.array([]) for _ in range(num_experts)]
    # subprocess.run(['sudo', 'mv', f'./MoeStackedBar_plot_{label}_100k.png', f'/moe-interpretability-pv/moe_stacked_bars_100k_data/MoeStackedBar_plot_{label}_100k.png'])
    # data_0_to_1000_jet_type_Higgs BB_expert_0_stacked_MoE_bars_100k.npy