from typing import Optional, Callable, Sequence
from math import log

import numpy as np
import torch as T
import torch.nn.functional as F

from myrl.agent.SACAgent import SACAgent
from myrl.agent.action import Actions, ObservationInput
from myrl.algorithm.base import Algorithm
from myrl.condition.learn_condition import LearnCondition, EveryLearn
from myrl.env.base import Environment
from myrl.memory.base import AbstractReplayBuffer
from myrl.model.optimizer import Optimizer
from myrl.spec.buffer import ValueEnum, ValueSpec, ValueStorageStrategyEnum
from myrl.stats.metrics import MetricEnum
from myrl.utils.algorithm import evaluate_agent
from myrl.utils.convertion import np_to_tensor, unpack_dict, unscale_continuous_action


class SAC(Algorithm):
    def __init__(self,
                 name: str,
                 agent: SACAgent,
                 env: Environment,
                 buffer: AbstractReplayBuffer,
                 start_learn_after_condition: LearnCondition,
                 learn_condition: LearnCondition, learning_end_condition: LearnCondition,
                 n_updates: int,
                 batch_size: int,
                 gamma: float = 0.99,
                 alpha: float = 1.0,
                 learn_alpha: bool = True,
                 target_entropy: Optional[float] = None,
                 alpha_optimizer_creator: Optional[Callable[[Sequence[T.Tensor]], Optimizer]] = None,
                 sync_q_target_every_update: int = 1):
        super().__init__(name, agent, env, buffer, start_learn_after_condition, learn_condition, learning_end_condition)
        self._gamma = gamma
        self._n_updates = n_updates
        self._batch_size = batch_size
        self._learn_alpha = learn_alpha
        self._update_counter = 0
        self._sync_q_target_every_update = sync_q_target_every_update
        if learn_alpha:
            if target_entropy is None:
                self._target_entropy = -np.prod(env.action_spec.shape).astype(np.float32)
            else:
                self._target_entropy = target_entropy
            self._log_alpha = T.tensor([log(alpha)], requires_grad=True)
            self._alpha_optimizer = alpha_optimizer_creator((self._log_alpha,))
            self._alpha = None
        else:
            assert target_entropy is None and alpha_optimizer_creator is None, \
                "target_entropy and alpha_optimizer_creator should be None in case of learn_alpha is False"
            self._alpha = T.tensor(alpha)
            self._target_entropy = None
            self._log_alpha = None
            self._alpha_optimizer = None

    def _preprocess_values_before_adding_to_buffer(self, values):
        actions = values.get(ValueEnum.ACTION.name)
        if self._action_spec.is_continuous and actions is not None:
            unscaled_actions = unscale_continuous_action(actions, self._action_spec)
            values[ValueEnum.ACTION.name] = unscaled_actions
        dones, infos = values.get(ValueEnum.DONE.name), values.get(ValueEnum.INFO.name)
        if dones is not None and infos is not None:
            dones = dones.copy()
            for i, info in enumerate(infos):
                is_truncated = info.get('TimeLimit.truncated', False)
                if is_truncated:
                    dones[i] = dones[i] * (1-is_truncated)
            values[ValueEnum.DONE.name] = dones
        return values

    def learn(self):
        super().learn()
        training_stats = []
        for i in range(self._n_updates):
            self._update_counter += 1
            sample = self._buffer.sample(self._batch_size, with_duplicates=True)
            values = np_to_tensor(sample.values)
            lengths = np_to_tensor(sample.lengths)
            (observations, actions, rewards, next_observations, dones), internal_state = unpack_dict(values, (
                ValueEnum.OBSERVATION.name,
                ValueEnum.ACTION.name,
                ValueEnum.REWARD.name,
                ValueEnum.NEXT_OBSERVATION.name,
                ValueEnum.DONE.name
            ))

            obs_input = ObservationInput(observations, lengths)
            obs_representation, current_internal_state = self._agent.representation(obs_input, internal_state)
            actions_on_policy = self._agent.policy(obs_representation)
            actions_on_policy.action(reparam=True, scale=False)
            if not self._action_spec.is_continuous:
                actions_on_policy = actions_on_policy.to_one_hot(self._action_spec.n_actions)

            if self._learn_alpha:
                alpha = T.exp(self._log_alpha.detach())
                alpha_loss = -(self._log_alpha * (actions_on_policy.log_prob() + self._target_entropy).detach()).mean()
                self._alpha_optimizer.optimize(alpha_loss, inputs=[self._log_alpha])
            else:
                alpha = self._alpha

            with evaluate_agent(self._agent):
                next_obs_input = ObservationInput(next_observations, lengths)
                next_obs_representation = self._agent.representation_target(next_obs_input, current_internal_state)
                next_actions = self._agent.policy(next_obs_representation)
                next_actions.action(reparam=True, scale=False)
                if not self._action_spec.is_continuous:
                    next_actions = next_actions.to_one_hot(self._action_spec.n_actions)

                q_targets = self._agent.q_targets(next_obs_representation, next_actions)
                min_q_target = T.min(q_targets, dim=0).values
                q_target = rewards + self._gamma * (1 - dones) * (
                            min_q_target - alpha * next_actions.log_prob().view(min_q_target.size()))

            if self._action_spec.is_continuous:
                actions = Actions(action=actions)
            else:
                actions = Actions(action=actions.squeeze(-1))
                actions = actions.to_one_hot(self._action_spec.n_actions)
            q_values, _ = self._agent.q_values(obs_representation.detach(), actions)
            critic_loss = 0.5 * sum(F.mse_loss(q_value, q_target) for q_value in q_values)

            q_values, _ = self._agent.q_values(obs_representation.detach(), actions_on_policy, {})
            min_q_value = T.min(q_values, dim=0).values
            policy_loss = (alpha * actions_on_policy.log_prob().view(min_q_value.size()) - min_q_value).mean()

            loss = policy_loss + critic_loss
            self._agent.zero_grad()
            loss.backward()
            self._agent.learning_step()

            if self._update_counter % self._sync_q_target_every_update == 0:
                self._agent.sync_q()

            training_stats.append({
                MetricEnum.LOSS: loss.detach().cpu().numpy(),
                MetricEnum.POLICY_LOSS: policy_loss.detach().cpu().numpy(),
                MetricEnum.CRITIC_LOSS: critic_loss.detach().cpu().numpy()
            })

    def get_buffer_value_specs(self) -> Sequence[ValueSpec]:
        return [
            ValueSpec(ValueEnum.OBSERVATION, self._observation_spec.shape),
            ValueSpec(ValueEnum.ACTION, self._action_spec.shape),
            ValueSpec(ValueEnum.REWARD, (1,)),
            ValueSpec(ValueEnum.DONE, (1,)),
            ValueSpec(ValueEnum.NEXT_OBSERVATION, self._observation_spec.shape,
                      storage_strategy=[ValueStorageStrategyEnum.EACH]),
        ]
