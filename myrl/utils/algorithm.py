from contextlib import contextmanager
from typing import Sequence, Any, Dict

import numpy as np
import torch as T
from torch import nn

from myrl.agent.base import Agent
from myrl.spec.buffer import ValueSpec

EMPTY_DICT = {}


def check_all_mandatory_values_specified(values: Dict[str, Any], value_specs: Sequence[ValueSpec]):
    for value_spec in value_specs:
        value_name = value_spec.name_str
        if value_spec.mandatory and value_name not in values:
            raise RuntimeError(f"{value_name}' value is mandatory, but was not found in passed values")


@contextmanager
def evaluate_agent(agent: Agent):
    with T.no_grad():
        is_training = agent.is_training
        agent.eval()
        try:
            yield agent
        finally:
            if is_training:
                agent.train()


def generalized_advantage_estimation(values: np.array,
                                     last_value: np.array,
                                     rewards: np.array,
                                     dones: np.array,
                                     gamma: float,
                                     gae_lambda: float,
                                     normalize_advantage: bool = False):
    """
    Calculates and returns GAE. The expected shapes for values, rewards, dones is [n_envs, N, *].
    :param values: Results of value function
    :param last_value: Result of value function on last seen next_observation. Shape is [n_envs, 1]
    :param rewards:
    :param dones:
    :param gamma: Reward discount factor
    :param gae_lambda: GAE regularization
    :param normalize_advantage: Whether to normalize advantages
    :return: Calculated advantages of shape [n_envs, N, 1]
    """
    assert 0 <= gamma <= 1 and 0 <= gae_lambda <= 1
    assert values.shape[0] == rewards.shape[0] == dones.shape[0]
    assert values.shape[1] == rewards.shape[1] == dones.shape[1]
    advantages = np.zeros((*values.shape[:2], 1), dtype=values.dtype)
    last_advantage = 0
    for t in reversed(range(values.shape[1])):
        episode_end_mask = 1.0 - dones[:, t]
        last_value = last_value * episode_end_mask
        last_advantage = last_advantage * episode_end_mask
        delta = rewards[:, t] + gamma * last_value - values[:, t]
        last_advantage = delta + gamma * last_advantage
        advantages[:, t] = last_advantage
        last_value = values[:, t]
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
    return advantages


def hard_sync(model: nn.Module, target: nn.Module):
    target.load_state_dict(model.state_dict())


def soft_sync(model: nn.Module, target: nn.Module, tau: float):
    assert 0.0 <= tau <= 1.0
    with T.no_grad():
        for source_param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data.copy_(source_param.data * tau + target_param.data * (1.-tau))
