"""This module implements a Vectorized Sampler used for OffPolicy Algorithms.

It diffs from OnPolicyVectorizedSampler in two parts:
 - The num of envs is defined by rollout_batch_size. In
 OnPolicyVectorizedSampler, the number of envs can be decided by batch_size
 and max_path_length. But OffPolicy algorithms usually samples transitions
 from replay buffer, which only has buffer_batch_size.
 - It needs to add transitions to replay buffer throughout the rollout.
"""
import itertools
import pickle

import numpy as np

from deepmdp.garage_mod.vec_env_executor import VecEnvExecutor

from garage.experiment import deterministic
from garage.misc import tensor_utils
from garage.sampler.batch_sampler import BatchSampler


class OffPolicyVectorizedSampler(BatchSampler):
    """This class implements OffPolicyVectorizedSampler.

    Args:
        algo (garage.np.RLAlgorithm): Algorithm.
        env (garage.envs.GarageEnv): Environment.
        n_envs (int): Number of parallel environments managed by sampler.
        no_reset (bool): Reset environment between samples or not.

    """

    def __init__(self, algo, env, n_envs=None, no_reset=True):
        if n_envs is None:
            n_envs = int(algo.rollout_batch_size)
        super().__init__(algo, env)
        self.n_envs = n_envs
        self.no_reset = no_reset

        self._last_obses = None
        self._last_q_vals = {i : [] for i in range(n_envs)}
        self._last_uncounted_discount = [0] * n_envs
        self._last_running_length = [0] * n_envs
        self._last_success_count = [0] * n_envs
        self.env_spec = self.env.spec
        self.vec_env = None

    def start_worker(self):
        """Initialize the sampler."""
        n_envs = self.n_envs
        envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(n_envs)]

        # Deterministically set environment seeds based on the global seed.
        for (i, e) in enumerate(envs):
            e.seed(deterministic.get_seed() + i)

        self.vec_env = VecEnvExecutor(
            envs=envs, max_path_length=self.algo.max_path_length)

    def shutdown_worker(self):
        """Terminate workers if necessary."""
        self.vec_env.close()

    # pylint: disable=arguments-differ, too-many-statements, too-many-branches
    def obtain_samples(self, itr, batch_size):
        """Collect samples for the given iteration number.

        Args:
            itr(int): Iteration number.
            batch_size(int): Number of environment interactions in one batch.

        Returns:
            list: A list of paths.

        """
        paths = []
        if not self.no_reset or self._last_obses is None:
            obses = self.vec_env.reset()
        else:
            obses = self._last_obses
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs
        n_samples = 0

        policy = self.algo.policy
        if self.algo.es:
            self.algo.es.reset()

        while n_samples < batch_size:
            policy.reset(dones)
            if self.algo.input_include_goal:
                obs = [obs['observation'] for obs in obses]
                d_g = [obs['desired_goal'] for obs in obses]
                a_g = [obs['achieved_goal'] for obs in obses]
                input_obses = np.concatenate((obs, d_g), axis=-1)
            else:
                input_obses = obses
            obs_normalized = tensor_utils.normalize_pixel_batch(
                self.env_spec, input_obses)
            if self.algo.es:
                actions, agent_infos = self.algo.es.get_actions(
                    itr, obs_normalized, self.algo.policy)
            else:
                actions, agent_infos = self.algo.policy.get_actions(
                    obs_normalized)

            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            new_episode_obs = None
            if "reset_new_obs" in env_infos:
                new_episode_obs = next_obses.copy()
                for i, reset_new_obs in env_infos["reset_new_obs"][0]:
                    new_episode_obs[i] = reset_new_obs
                del env_infos["reset_new_obs"]

            #self.vec_env.envs[0].render()

            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            n_samples += len(next_obses)

            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]

            if self.algo.input_include_goal:
                self.algo.replay_buffer.add_transitions(
                    observation=obs,
                    action=actions,
                    goal=d_g,
                    achieved_goal=a_g,
                    terminal=dones,
                    next_observation=[
                        next_obs['observation'] for next_obs in next_obses
                    ],
                    next_achieved_goal=[
                        next_obs['achieved_goal'] for next_obs in next_obses
                    ],
                )
            else:
                payload = {
                    "observation": obses,
                    "action": actions,
                    "reward": rewards * self.algo.reward_scale,
                    "terminal": dones,
                    "next_observation": next_obses
                }
                if env_infos and env_infos[0].get("ground_truth_state") is not None:
                    payload["ground_truth_state"] = [env_info.get("ground_truth_state") for env_info in env_infos]

                self.algo.replay_buffer.add_transitions(
                    **payload
                )

            for idx, reward, env_info, q_val, done in zip(itertools.count(), rewards,
                                                   env_infos, agent_infos["q_vals"], dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        rewards=[],
                        env_infos=[],
                        dones=[],
                        q_vals=self._last_q_vals[idx].copy(),
                        undiscounted_return=self._last_uncounted_discount[idx],
                        # running_length: Length of path up to now
                        # Note that running_length is not len(rewards)
                        # Because a path may not be complete in one batch
                        running_length=self._last_running_length[idx],
                        success_count=self._last_success_count[idx])

                running_paths[idx]['rewards'].append(reward)
                running_paths[idx]['env_infos'].append(env_info)
                running_paths[idx]['dones'].append(done)
                running_paths[idx]['q_vals'].append(q_val)
                running_paths[idx]['running_length'] += 1
                running_paths[idx]['undiscounted_return'] += reward
                running_paths[idx]['success_count'] += env_info.get(
                    'is_success') or 0

                self._last_q_vals[idx].append(q_val)
                self._last_uncounted_discount[idx] += reward
                self._last_success_count[idx] += env_info.get(
                    'is_success') or 0
                self._last_running_length[idx] += 1

                if done or n_samples >= batch_size:
                    paths.append(
                        dict(
                            rewards=np.asarray(running_paths[idx]['rewards']),
                            dones=np.asarray(running_paths[idx]['dones']),
                            env_infos=tensor_utils.stack_tensor_dict_list(
                                running_paths[idx]['env_infos']),
                            q_vals=np.asarray(running_paths[idx]["q_vals"]),
                            running_length=running_paths[idx]
                            ['running_length'],
                            undiscounted_return=running_paths[idx]
                            ['undiscounted_return'],
                            success_count=running_paths[idx]['success_count']))
                    running_paths[idx] = None

                    if done:
                        self._last_q_vals[idx] = []
                        self._last_running_length[idx] = 0
                        self._last_success_count[idx] = 0
                        self._last_uncounted_discount[idx] = 0

                    if self.algo.es:
                        self.algo.es.reset()
            if new_episode_obs:
                obses = new_episode_obs
            else:
                obses = next_obses
            self._last_obses = obses
        return paths
