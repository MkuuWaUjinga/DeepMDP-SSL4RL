#!/bin/bash
#SBATCH --job-name=deepmdp-repro
#SBATCH --time=0
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --nodelist=simba-compute-gpu-1
srun hostname
srun python scripts/experiment_dqn_baseline.py --config_path scripts/configs/LunarLander/obf-state-obs/lunar_lander_deepmdp.json 
