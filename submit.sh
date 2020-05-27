#!/bin/bash
#SBATCH --job-name=deepmdp-repro
srun hostname
srun python experiment_dqn_baseline.py