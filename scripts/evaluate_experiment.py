import pickle
from deepmdp.garage_mod.policies.discrete_qf_derived_policy import DiscreteQfDerivedPolicy
from garage.envs import GarageEnv

experiment_run_number = 452
number_of_episodes = 100
with open("runs/snapshots/params.pkl", "rb") as file:
    exp = pickle.load(file)
env = GarageEnv(env_name="LunarLander-v2")
algo = exp['algo']
policy = DiscreteQfDerivedPolicy(
    env.spec,
    algo.qf
)

for _ in range(number_of_episodes):
    obs = env.reset()
    env.render()
    done = False
    while not done:
        act, _ = policy.get_action(obs)
        print(act.numpy())
        obs, reward, done, _ = env.step(act.numpy())
        print(done)
        env.render()