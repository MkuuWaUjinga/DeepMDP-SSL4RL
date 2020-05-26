#!/bin/bash
#SBATCH --job-name=deepmdp-repro
srun hostname
srun python deepmdp-repro/scripts/experiment_dqn_baseline.py