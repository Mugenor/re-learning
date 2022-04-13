import numpy as np

from myrl.algorithm.PPO import PPO
from myrl.agent.PolicyValueAgent import PolicyValueSharedAgent
from myrl.condition.learn_condition import EverySteps, AfterLearnsDone
from myrl.env.GymEnvironment import GymEnvironment
from myrl.env.wrapper.velocity import VelocityMaskWrapper
from myrl.memory.recurrent.buffer import RecurrentReplayBuffer
from myrl.model.ppo_simple import PPOSimpleRecurrentPolicyValueSharedModel
from myrl.model.optimizer import Optimizer

import torch.optim as optim
import torch.nn as nn

from myrl.spec.buffer import ValueEnum
from myrl.utils.algorithm import EMPTY_DICT


def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window


if __name__ == '__main__':
    env = GymEnvironment('CartPole-v1', n_envs=16, multiprocess=False, wrappers=[VelocityMaskWrapper])
    model = PPOSimpleRecurrentPolicyValueSharedModel(env.observation_spec, env.action_spec)
    model_optimzier = Optimizer(
        optim.AdamW(model.parameters(), lr=3e-4),
        pre_step_hooks=[lambda: nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)]
    )
    agent = PolicyValueSharedAgent(
        env.action_spec, model, model_optimzier
    )
    buffer = RecurrentReplayBuffer(4096, max_sequence_length=8, is_circular=False)
    algorithm = PPO(
        name='PPO',
        agent=agent,
        env=env,
        buffer=buffer,
        learn_condition=EverySteps(256),
        learning_end_condition=AfterLearnsDone(100),

        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=False,
        clip_range=0.2,
        entropy_bonus_coef=0.001,
        value_loss_coef=0.2,
        epochs=4,
        batch_size=128
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
        if step_counter % 100 == 0:
            rewards_slice = all_rewards[-50:]
            print(f'{step_counter}: {round(np.mean(rewards_slice), 3)} +- {round(np.std(rewards_slice), 3)}')
