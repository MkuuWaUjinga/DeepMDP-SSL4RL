#!/bin/bash
#SBATCH --job-name=deepmdp-repro
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=simba-compute-11
srun hostname
srun python scripts/experiment_dqn_baseline.py
