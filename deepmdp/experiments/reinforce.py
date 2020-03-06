import time
import sacred
import gym
import torch
import os

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import LocalRunner, SnapshotConfig
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.algos import VPG
from garage.torch.policies import DeterministicMLPPolicy

ex = sacred.experiment.Experiment("Reinforce-Baseline")


@ex.config
def config():
    snapshot_config = {"snapshot_dir": (os.path.join(os.getcwd(), 'runs/snapshots')),
                       "snapshot_mode": "last",
                       "snapshot_gap": 1}
    env_name = "SpaceInvaders-v0"


def run_task(snapshot_config, env_name):
    runner = LocalRunner(snapshot_config)
    env = GarageEnv(normalize(gym.make(env_name)))
    policy = DeterministicMLPPolicy(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=torch.relu,
                                    output_nonlinearity=torch.tanh)
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    vpg = VPG(env.spec,
              policy,
              baseline,
              optimizer=torch.optim.Adam)

    # VPG takes as default BatchSampler that has n_envs as required positional arg...
    runner.setup(algo=vpg, env=env, sampler_args={"n_envs":1})
    runner.train(n_epochs=400, batch_size=128)


@ex.main
def run(snapshot_config, env_name):
    snapshot_config = SnapshotConfig(snapshot_config["snapshot_dir"],
                                     snapshot_config["snapshot_mode"],
                                     snapshot_config["snapshot_gap"])
    run_task(snapshot_config, env_name)
