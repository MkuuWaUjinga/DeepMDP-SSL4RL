#!/usr/bin/env bash
xvfb-run --server-args="-screen 0 1000x1000x24" python experiment_dqn_baseline.py --config_path configs/LunarLander/image-obs/lunar_lander.json