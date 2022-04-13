import random
import numpy as np
import torch as T
from torch import optim, nn

from myrl.agent.SACAgent import SACAgent
from myrl.agent.action import Actions
from myrl.algorithm.SAC import SAC
from myrl.condition.learn_condition import EverySteps, AfterLearnsDone, AfterEpisodesCompleted, AfterStepsDone
from myrl.env.GymEnvironment import GymEnvironment
from myrl.memory.ReplayBuffer import ReplayBuffer
from myrl.model.optimizer import Optimizer
from myrl.model.sac_simple import SACSimpleQValueFunction, SACGaussianPolicy
from myrl.model.twin import SoftTargetTwinModel
from myrl.spec.buffer import ValueEnum
from myrl.utils.algorithm import EMPTY_DICT
from myrl.utils.convertion import unscale_continuous_action


def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    env = GymEnvironment('Pendulum-v0', n_envs=6, multiprocess=False, seed=seed)
    policy = SACGaussianPolicy(env.observation_spec, env.action_spec,
                               hidden1=256, hidden2=256,
                               log_std_min=-20, log_std_max=2,
                               activation_fn_creator=nn.ReLU)
    policy_optimizer = Optimizer(
        optim.Adam(policy.parameters(), lr=0.0003),
        # pre_step_hooks=[lambda: nn.utils.clip_grad_norm_(policy.parameters(), 1.0)]
    )
    q_functions = [SoftTargetTwinModel(SACSimpleQValueFunction(env.observation_spec,
                                                               env.action_spec,
                                                               hidden=256,
                                                               activation_fn_creator=nn.ReLU),
                                       tau=0.005)
                   for _ in range(2)]
    q_function_optimizers = [
        Optimizer(
            optim.Adam(q_function.parameters(), lr=0.0003),
            # pre_step_hooks=[lambda: nn.utils.clip_grad_norm_(q_function.parameters(), 1.0)]
        ) for q_function in q_functions
    ]
    agent = SACAgent(
        env.action_spec,
        policy, policy_optimizer,
        q_functions, q_function_optimizers
    )
    # buffer = ReplayBuffer(16)
    # steps_as_random_sample = 4
    buffer = ReplayBuffer(1_000_000, is_circular=True)
    steps_as_random_sample = 100
    algorithm = SAC(
        name='SAC',
        agent=agent,
        env=env,
        buffer=buffer,
        start_learn_after_condition=AfterStepsDone(100),
        # start_learn_after_condition=AfterStepsDone(4),
        learn_condition=EverySteps(1),
        learning_end_condition=AfterStepsDone(15_000),

        gamma=0.99,
        batch_size=256,
        # batch_size=6,
        n_updates=1,
        learn_alpha=True,
        alpha_optimizer_creator=lambda alpha_params: Optimizer(optim.Adam(alpha_params, lr=0.0003)),
        sync_q_target_every_update=1
    ).train()

    obs = env.reset()
    internal_state = EMPTY_DICT
    episode_reward = np.zeros(env.n_envs)
    all_rewards = []
    step_counter = 0
    print('Start learning')
    while algorithm.should_continue_learning:
        if step_counter < steps_as_random_sample:
            actions = Actions(action=T.tensor([env.env.single_action_space.sample() for _ in range(env.n_envs)]))
        else:
            actions, internal_state = algorithm.act(obs, internal_state)
        next_obs, rewards, dones, infos = env.step(actions.np_action())
        values = {
            ValueEnum.OBSERVATION.name: obs,
            ValueEnum.ACTION.name: actions.np_action().reshape(env.n_envs, -1),
            ValueEnum.REWARD.name: rewards.reshape(env.n_envs, -1),
            ValueEnum.DONE.name: dones.reshape(env.n_envs, -1),
            ValueEnum.NEXT_OBSERVATION.name: next_obs,
            ValueEnum.INFO.name: infos
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
            print(f'{step_counter}: {round(np.mean(rewards_slice), 3)} +- {round(np.std(rewards_slice), 3)}, the last one: {all_rewards[-1]}')

