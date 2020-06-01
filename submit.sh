#!/bin/bash
#SBATCH --job-name=deepmdp-repro
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --nodelist=simba-compute-gpu-3
srun hostname
srun python scripts/experiment_dqn_baseline.py --config-path scripts/configs/LunarLander/scalar-obs/baseline-new-architecture/lunar_lander_deepmdp.json 
