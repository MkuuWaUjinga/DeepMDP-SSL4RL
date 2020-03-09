import sacred
import gym
import torch
import os

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import LocalRunner, SnapshotConfig
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.algos import VPG
from garage.torch.policies import GaussianMLPPolicy

ex = sacred.experiment.Experiment("Reinforce-Baseline")


@ex.config
def config():
    snapshot_config = {"snapshot_dir": (os.path.join(os.getcwd(), 'runs/snapshots')),
                       "snapshot_mode": "last",
                       "snapshot_gap": 1}
    env_name = "SpaceInvaders-v0"


def run_task(snapshot_config, *_):
    runner = LocalRunner(snapshot_config)
    env = GarageEnv(normalize(gym.make("SpaceInvaders-v0")))
    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    print(env.spec.observation_space.flat_dim)
    print(env.spec.action_space.flat_dim)
    print(policy._mean_module)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = VPG(env_spec=env.spec,
               policy=policy,
               optimizer=torch.optim.Adam,
               baseline=baseline,
               max_path_length=100,
               discount=0.99,
               center_adv=True,
               policy_lr=1e-2)

    runner.setup(algo=algo, env=env)
    runner.train(n_epochs=400, batch_size=100)


@ex.main
def run(snapshot_config, env_name):
    snapshot_config = SnapshotConfig(snapshot_config["snapshot_dir"],
                                     snapshot_config["snapshot_mode"],
                                     snapshot_config["snapshot_gap"])
    run_task(snapshot_config, env_name)
