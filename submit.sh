#!/bin/bash
#SBATCH --job-name=deepmdp-repro
#SBATCH --time=05:00:00
#SBATCH --ntasks=8
srun hostname
srun python scripts/experiment_dqn_baseline.py