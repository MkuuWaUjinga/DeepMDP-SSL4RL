#!/bin/bash
#SBATCH --job-name=deepmdp-repro
srun hostname
srun visdom -port 9098
srun python deepmdp-repro/scripts/experiment_dqn_baseline.py