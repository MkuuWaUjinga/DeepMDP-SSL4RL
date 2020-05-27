#!/bin/bash
#SBATCH --job-name=deepmdp-repro
srun hostname
srun python scripts/experiment_dqn_baseline.py