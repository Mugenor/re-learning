import pybullet_envs
import numpy as np
from torch import optim, nn
import torch as T

from myrl.agent.SACAgent import SACAgent
from myrl.algorithm.SAC import SAC
from myrl.condition.learn_condition import EverySteps, AfterLearnsDone, AfterEpisodesCompleted
from myrl.env.GymEnvironment import GymEnvironment
from myrl.memory.ReplayBuffer import ReplayBuffer
from myrl.model.optimizer import Optimizer
from myrl.model.sac_simple import SACGumbelSoftmaxPolicy, SACSimpleQValueFunction, SACGaussianPolicy
from myrl.model.twin import SoftTargetTwinModel
from myrl.spec.buffer import ValueEnum
from myrl.utils.algorithm import EMPTY_DICT


def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window


if __name__ == '__main__':
    env = GymEnvironment('LunarLanderContinuous-v2', n_envs=16, multiprocess=False)
    policy = SACGaussianPolicy(env.observation_spec, env.action_spec,
                               hidden1=256, hidden2=256,
                               log_std_min=-20, log_std_max=2)
    policy_optimizer = Optimizer(
        optim.Adam(policy.parameters(), lr=3e-4),
        pre_step_hooks=[lambda: nn.utils.clip_grad_norm_(policy.parameters(), 1.0)]
    )
    q_functions = [SoftTargetTwinModel(SACSimpleQValueFunction(env.observation_spec,
                                                               env.action_spec,
                                                               hidden=256),
                                       tau=0.005)
                   for _ in range(2)]
    q_function_optimizers = [
        Optimizer(
            optim.Adam(q_function.parameters(), lr=3e-4),
            pre_step_hooks=[lambda: nn.utils.clip_grad_norm_(q_function.parameters(), 1.0)]
        ) for q_function in q_functions
    ]
    agent = SACAgent(
        env.action_spec,
        policy, policy_optimizer,
        q_functions, q_function_optimizers
    )
    buffer = ReplayBuffer(1_000_000, is_circular=True)
    algorithm = SAC(
        name='SAC',
        agent=agent,
        env=env,
        buffer=buffer,
        learn_condition=EverySteps(100),
        learning_end_condition=AfterLearnsDone(1000),

        gamma=0.99,
        batch_size=256,
        n_updates=4,
        learn_alpha=True,
        alpha_optimizer_creator=lambda alpha_params: Optimizer(optim.Adam(alpha_params, lr=0.001)),
        sync_q_target_every_update=2
    ).train()

    obs = env.reset()
    internal_state = EMPTY_DICT
    episode_reward = np.zeros(env.n_envs)
    all_rewards = []
    step_counter = 0
    while algorithm.should_continue_learning:
        actions, internal_state = algorithm.act(obs, internal_state)
        next_obs, rewards, dones, infos = env.step(actions.np_action())
        values = {
            ValueEnum.OBSERVATION.name: obs,
            ValueEnum.ACTION.name: actions.np_action().reshape(env.n_envs, -1),
            ValueEnum.REWARD.name: rewards.reshape(env.n_envs, -1) / 100.0,
            ValueEnum.DONE.name: dones.reshape(env.n_envs, -1),
            ValueEnum.NEXT_OBSERVATION.name: next_obs
        }
        values.update(internal_state)
        algorithm.observe(values)
        obs = next_obs
        step_counter += 1
        episode_reward += rewards
        if dones.any():
            reset_obs = env.reset(dones)
            reset_indices = np.nonzero(dones)[0]
            obs[reset_indices] = reset_obs
            all_rewards.extend(episode_reward[reset_indices])
            episode_reward[reset_indices] = 0.
        if step_counter % 50 == 0 and len(all_rewards) > 0:
            rewards_slice = all_rewards[-50:]
            print(f'{step_counter}: {round(np.mean(rewards_slice), 3)} +- {round(np.std(rewards_slice), 3)}')
            print(f'Episode ended: {len(all_rewards)}')

