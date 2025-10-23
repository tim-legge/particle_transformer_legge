# MoE Particle Transformer Interpretability Project

This repository implements and analyzes Mixture of Experts (MoE) applied to particle physics jet classification using Particle Transformers, with a focus on interpretability research.

## Architecture Overview

**Core Model**: `MoeParticleTransformer.py` contains a custom implementation where each transformer block uses MoE in the feed-forward layers instead of standard dense layers. Key components:

- **Router**: Linear layer that routes tokens to experts based on learned gating weights
- **Experts**: Independent feed-forward networks (4 experts by default, configurable via `moe_num_experts`)
- **Top-k Routing**: Supports both top-1 (Switch-style) and top-k routing (`moe_top_k` parameter)
- **Load Balancing**: Auxiliary loss prevents expert collapse (`moe_aux_loss_coef=0.01`)

**Data Pipeline**: Uses particle physics datasets (JetClass, QuarkGluon, TopLandscape) with specialized preprocessing:
- Particle features: `(px, py, pz, energy)` → `(pt, rapidity, phi, mass)` coordinates
- Pairwise features: Delta-R, kt, z, invariant mass between particle pairs
- Feature engineering in `dataloader.py` handles ROOT files and awkward arrays

## Key Patterns

### MoE Configuration
```python
# In Block.__init__()
moe_num_experts=4      # Number of expert networks
moe_top_k=1           # How many experts to route each token to  
moe_capacity_factor=1.25  # Expert capacity buffer
moe_aux_loss_coef=0.01    # Load balancing loss weight
moe_router_jitter=0.0     # Training noise for exploration
```

### Router Analysis Workflow
The notebook `MoeRouterHooks.ipynb` demonstrates the interpretability approach:
1. **Hook Installation**: Capture router outputs (`topk_idx`, `topk_w`) during inference
2. **Expert Specialization**: Analyze which experts activate for different particle types
3. **Routing Patterns**: Visualize expert selection across different jet types

### Training Scripts
Use dataset-specific training scripts with the `weaver` framework:
```bash
# JetClass dataset with Particle Transformer
./train_JetClass.sh ParT full
# Supports: "kin" (kinematic), "kinpid" (kinematic+PID), "full" features
```

### Data Configuration
YAML configs in `data/` define:
- **Input features**: Which particle/jet features to use
- **Preprocessing**: Log transforms, standardization, derived features
- **Data loading**: Padding, masking, batch configuration

## Development Workflow

### Model Modifications
- Modify `MoeParticleTransformer.py` for architectural changes
- Router logic in `Block.forward()` - supports both top-1 and top-k routing
- Expert networks are simple MLPs, easily extensible

### Analysis Tools
- Use `MoeRouterHooks.ipynb` for interpretability experiments
- `dataloader.py` provides utilities for dataset manipulation
- Pretrained models in `models/` directory for different feature sets

### Environment Setup
```bash
source env.sh  # Set dataset paths
# Configure DATADIR_JetClass, DATADIR_TopLandscape, DATADIR_QuarkGluon
```

### Dependencies
- PyTorch with CUDA support
- `weaver` framework for particle physics ML
- `awkward`, `uproot`, `vector` for particle physics data handling
- Standard ML stack: numpy, matplotlib for analysis

## Important Notes

- **Auxiliary Loss**: Always include MoE auxiliary loss in total training loss to prevent expert collapse
- **Capacity Factor**: Increase if seeing tokens dropped during training (check router utilization)
- **Memory**: MoE increases memory usage ~linearly with number of experts
- **Particle Physics Context**: Features like `delta_phi`, `delta_r2` have special physics meaning for particle separation
- **Coordinate Systems**: The code handles conversions between Cartesian `(px,py,pz,E)` and physics coordinates `(pt,eta,phi,m)`

## Common Patterns

- **Feature Engineering**: Pairwise particle relationships are crucial (see `pairwise_lv_fts()`)
- **Masking**: Real particles vs padding handled throughout pipeline
- **Sequence Trimming**: Dynamic sequence length handling in `SequenceTrimmer` for efficiency
- **Physics Constraints**: Lorentz-invariant features and relativistic calculations preserved