# Flow-Based Vision-Conditioned Robotic Manipulation

This repository implements **Conditional Flow Matching (CFM)** and **Mean Flow Matching (MFM)** for robotic pushing tasks using the **PushT dataset**. Both models learn vision-conditioned trajectory policies.

## Overview

- **CFM (Conditional Flow Matching)**: Stochastic model generating diverse trajectories.
- **MFM (Mean Flow Matching)**: Deterministic model predicting mean trajectories for stable execution.
- Evaluated on trajectory coverage metrics over 4000 epochs.

## Repository Structure

- flow_matching/: Core code (train.py, evaluate.py, Checkpoints/)
- data/: PushT dataset
- Images/: Plots and diagrams
- main.ipynb: Example notebook
- requirements.txt: Python dependencies
- README.md

## Installation

1. Clone the repo:

2. Install dependencies:

pip install -r requirements.txt

3. Download the PushT dataset and place it in `data/`  
[PushT Dataset](https://arxiv.org/abs/2307.00429)


**Key parameters:**  
- Epochs: 4000  
- Batch size: 512  
- Optimizer: AdamW (LR=1e-4)  
- Scheduler: Cosine decay  
- EMA: Yes, power=0.75  

## Evaluation

Metrics: Maximum coverage, Mean coverage, Std deviation, Zero-coverage (%)  


## Results Summary

| Metric               | CFM | MFM |
|---------------------|-----|-----|
| Max Coverage         | 0.991 | 0.635 |
| Mean Coverage        | 0.406 | 0.069 |
| Std Deviation        | 0.385 | 0.118 |
| Zero-Coverage (%)    | 25–44 | 52–60 |
| Trajectory Diversity | High  | Low |
| Predictability       | Moderate/Low | High |

## Future Work

- Real robot experiments  
- Multi-object manipulation  
- Hybrid stochastic + deterministic flows  
- Long-horizon planning  
- Adaptive conditioning (force/tactile sensors)  
- Efficiency improvements  

## References

- PushT Dataset: https://arxiv.org/abs/2307.00429  
- Flow Matching: Lipman et al., NeurIPS 2023  
- Diffusion Models: Ho et al., NeurIPS 2020  
- Recent Generative Flows: https://arxiv.org/abs/2409.01083

