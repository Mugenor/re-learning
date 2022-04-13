from typing import Sequence, Tuple, Any, Dict

import numpy as np
import torch as T

from myrl.agent.PolicyValueAgent import AbstractPolicyValueAgent
from myrl.agent.action import Actions, ObservationInput
from myrl.algorithm.base import Algorithm
from myrl.condition.learn_condition import LearnCondition
from myrl.env.base import Environment
from myrl.memory.base import AbstractReplayBuffer
from myrl.spec.buffer import ValueEnum, ValueSpec
from myrl.stats.metrics import MetricEnum
from myrl.utils.algorithm import evaluate_agent, generalized_advantage_estimation
from myrl.utils.convertion import np_to_tensor, unpack_dict, create_sequence_mask


class PPO(Algorithm):
    def __init__(self,
                 name: str,
                 agent: AbstractPolicyValueAgent,
                 env: Environment,
                 buffer: AbstractReplayBuffer,
                 learn_condition: LearnCondition,
                 learning_end_condition: LearnCondition,

                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 normalize_advantage: bool = True,
                 clip_range: float = 0.2,
                 entropy_bonus_coef: float = 0.001,
                 value_loss_coef: float = 0.5,
                 epochs: int = 4,
                 batch_size: int = 64):
        super().__init__(name, agent, env, buffer, learn_condition, learning_end_condition)
        self._epochs = epochs
        self._batch_size = batch_size
        assert clip_range >= 0
        self._clip_range = clip_range
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._normalize_advantage = normalize_advantage
        self._entropy_bonus_coef = entropy_bonus_coef
        self._value_loss_coef = value_loss_coef

    def act(self, observations: np.array, internal_state: Dict[str, Any] = None) -> Tuple[Actions, Dict[str, Any]]:
        with evaluate_agent(self._agent):
            actions, values, internal_state = self._agent.action_value(
                ObservationInput(observations),
                internal_state
            )
            log_probs = actions.log_prob()
            # values to store in buffer in future. See get_buffer_value_specs method
            internal_state.update({
                ValueEnum.VALUE.name: values.reshape(observations.shape[0], 1),
                ValueEnum.LOG_PROB.name: log_probs.reshape(observations.shape[0], -1)
            })
            return actions, internal_state

    def learn(self):
        if len(self._buffer) < self._batch_size:
            return
        super().learn()
        data = self._buffer.all_data
        # shape is [N, L, *]
        (rewards, dones, values), _ = \
            unpack_dict(data.continuous_values_by_envs, (
                ValueEnum.REWARD.name,
                ValueEnum.DONE.name,
                ValueEnum.VALUE.name
            ))
        (last_next_observations,), last_internal_states = unpack_dict(self._buffer.last_values_per_env,
                                                                      (ValueEnum.NEXT_OBSERVATION.name,))
        advantages = self._calculate_advantage(values, last_next_observations, rewards, dones, last_internal_states)
        data.add_continuous(ValueEnum.ADVANTAGE.name, advantages)
        train_stats = []
        for epoch in range(self._epochs):
            for values, internal_states, lengths in data.iterate_batches(self._batch_size):
                values, lengths = np_to_tensor(values), np_to_tensor(lengths)
                internal_states = np_to_tensor(internal_states)
                (observations_, actions_, advantages_, log_probs_, values_), _ = unpack_dict(values, (
                    ValueEnum.OBSERVATION.name,
                    ValueEnum.ACTION.name,
                    ValueEnum.ADVANTAGE.name,
                    ValueEnum.LOG_PROB.name,
                    ValueEnum.VALUE.name
                ))
                sequences_mask = create_sequence_mask(lengths)

                if self._normalize_advantage:
                    normalized_advantage = (advantages_ - advantages_.mean()) / (advantages_.std() + 1e-8)
                else:
                    normalized_advantage = advantages_

                new_actions, new_values, _ = self._agent.action_value(
                    ObservationInput(observations_, lengths),
                    internal_states
                )
                new_log_probs = new_actions.log_prob(
                    actions_.reshape(new_actions.action().shape)
                ).reshape(actions_.shape)
                ratio = T.exp(new_log_probs - log_probs_)
                surr1 = ratio * normalized_advantage
                surr2 = T.clamp(ratio, 1 - self._clip_range, 1 + self._clip_range) * normalized_advantage
                policy_loss = T.min(surr1, surr2)[sequences_mask].mean()

                sampled_return = values_ + advantages_
                clipped_value = values_ + (new_values - values_).clamp(-self._clip_range, self._clip_range)
                value_loss = T.max((new_values-sampled_return) ** 2, (clipped_value-sampled_return) ** 2) \
                    [sequences_mask].mean()

                if self._entropy_bonus_coef is not None and self._entropy_bonus_coef > 0.:
                    entropy_bonus = new_actions.distribution.entropy()[sequences_mask].mean()
                else:
                    entropy_bonus = T.tensor(0.0)

                loss = -(policy_loss
                         + self._entropy_bonus_coef * entropy_bonus
                         - self._value_loss_coef * value_loss)
                self._agent.zero_grad()
                loss.backward()
                self._agent.learning_step()

                train_stats.append({
                    MetricEnum.LOSS: loss.detach().cpu().numpy(),
                    MetricEnum.POLICY_LOSS: policy_loss.detach().cpu().numpy(),
                    MetricEnum.VALUE_LOSS: value_loss.detach().cpu().numpy(),
                    MetricEnum.ENTROPY_BONUS: entropy_bonus.detach().cpu().numpy()
                })
        self._buffer.clear()

    def _calculate_advantage(self,
                             values: np.ndarray,
                             last_next_observations: np.ndarray,
                             rewards: np.ndarray,
                             dones: np.ndarray,
                             last_internal_states: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates GAE
        :param values: of shape [n_envs, N, 1]
        :param next_observations: of shape [n_envs, *]. Last observation per each env
        :param rewards: of shape [n_envs, N, 1]
        :param dones: of shape [n_envs, N, 1]
        :param internal_states: Dictionary of internal states, which will be passed to agent for calculating last value
        :return:
        """
        with evaluate_agent(self._agent):
            last_value = self._agent.value(
                ObservationInput(last_next_observations),
                np_to_tensor(last_internal_states)
            )[0].detach().cpu().numpy()
        advantages = generalized_advantage_estimation(
            values, last_value, rewards, dones,
            self._gamma, self._gae_lambda)
        return advantages

    def get_buffer_value_specs(self) -> Sequence[ValueSpec]:
        return super().get_buffer_value_specs() + [
            ValueSpec(ValueEnum.LOG_PROB, self._action_spec.shape),
            ValueSpec(ValueEnum.VALUE, (1,))
        ]
