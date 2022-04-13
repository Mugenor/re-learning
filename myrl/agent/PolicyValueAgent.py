from abc import abstractmethod
from typing import Dict, Any, Tuple

import torch as T

from myrl.agent.action import Actions, ObservationInput
from myrl.agent.base import Agent
from myrl.model.base import Policy, ValueFunction, PolicyValueSharedModel
from myrl.model.optimizer import Optimizer
from myrl.spec.buffer import ValueEnum, ValueSpec, ValueStorageStrategyEnum
from myrl.spec.env import ActionSpec


class AbstractPolicyValueAgent(Agent):
    def __init__(self,
                 action_spec: ActionSpec):
        super().__init__(action_spec)

    @abstractmethod
    def action_value(self,
                     observations: ObservationInput,
                     internal_state: Dict[str, Any] = None) -> Tuple[Actions, T.Tensor, Dict[str, Any]]:
        """
        Executes both: policy and value function and returns the result
        :param observations: Array of shape: [N, *OBS] or [N, L, *OBS],
            where N - number of observations, *OBS - dimensions of observations
        :param internal_state: Internal state of agent for specific observation. Useful for recurrent NNs
            Either None in case of agent has no internal state or dict of internal state keys to values of length N.
        :return: Tuple of actions, values of shape [N, 1] and internal states (or None in case of no internal state).
        """
        pass

    @abstractmethod
    def value(self,
              observations: ObservationInput,
              internal_state: Dict[str, Any] = None) -> Tuple[T.Tensor, Dict[str, Any]]:
        """
        Executes value function and returns the result
        :param observations: observations: Array of shape: [N, *OBS],
            where N - number of observations, *OBS - dimensions of observations
        :param internal_state: Internal state of agent for specific observation. Useful for recurrent NNs
            Either None in case of agent has no internal state or dict of internal state keys to values of length N.
        :return: Tuple of values of shape [N, 1] and internal states (or None in case of no internal state)
        """
        pass


class PolicyValueSeparatedAgent(AbstractPolicyValueAgent):
    def __init__(self,
                 action_spec: ActionSpec,

                 policy: Policy,
                 policy_optimizer: Optimizer,
                 value_function: ValueFunction,
                 value_fn_optimizer: Optimizer):
        super().__init__(action_spec)
        self._policy = policy
        self._policy_optimizer = policy_optimizer
        self._value_function = value_function
        self._value_fn_optimizer = value_fn_optimizer

    def action_value(self,
                     observations: ObservationInput,
                     internal_state: Dict[str, Any] = None) -> Tuple[Actions, T.Tensor, Dict[str, Any]]:
        next_internal_state = {}
        actions, next_policy_internal_state = self.act(observations, internal_state)
        values, next_value_internal_state = self.value(observations, internal_state)
        next_internal_state.update(next_policy_internal_state)
        next_internal_state.update(next_value_internal_state)
        return actions, values, next_internal_state

    def value(self, observations: ObservationInput, internal_state: Dict[str, Any] = None) -> Tuple[T.Tensor, Dict[str, Any]]:
        value_internal_state = internal_state.get(ValueEnum.VALUE_INTERNAL_STATE.name, None)
        values, next_value_internal_state = self._value_function(observations, value_internal_state)
        if next_value_internal_state is None:
            return values, {}
        else:
            return values, {ValueEnum.VALUE_INTERNAL_STATE.name: next_value_internal_state}

    def act(self, observations: ObservationInput, internal_state: Dict[str, Any] = None) -> Tuple[Actions, Dict[str, Any]]:
        policy_internal_state = internal_state.get(ValueEnum.POLICY_INTERNAL_STATE.name, None)
        actions, next_policy_internal_state = self._policy(observations, policy_internal_state)
        if next_policy_internal_state is None:
            return actions, {}
        else:
            return actions, {ValueEnum.POLICY_INTERNAL_STATE.name: next_policy_internal_state}

    def zero_grad(self):
        self._policy_optimizer.zero_grad()
        self._value_fn_optimizer.zero_grad()

    def learning_step(self):
        self._policy_optimizer.step()
        self._value_fn_optimizer.step()

    def train(self):
        super().train()
        self._policy.train()
        self._value_function.train()

    def eval(self):
        super().eval()
        self._policy.eval()
        self._value_function.eval()


class PolicyValueSharedAgent(AbstractPolicyValueAgent):
    def __init__(self,
                 action_spec: ActionSpec,

                 model: PolicyValueSharedModel,
                 model_optimizer: Optimizer):
        super().__init__(action_spec)
        self._model = model
        self._model_optimzier = model_optimizer

    def action_value(self,
                     observations: ObservationInput,
                     internal_state: Dict[str, Any] = None) -> Tuple[Actions, T.Tensor, Dict[str, Any]]:
        model_internal_state = internal_state.get(ValueEnum.INTERNAL_STATE.name, None)
        actions, values, next_internal_state = self._model(observations, model_internal_state)
        if next_internal_state is None:
            return actions, values, {}
        else:
            return actions, values, {ValueEnum.INTERNAL_STATE.name: next_internal_state}

    def value(self, observations: ObservationInput, internal_state: Dict[str, Any] = None) -> Tuple[T.Tensor, Dict[str, Any]]:
        _, values, next_internal_state = self.action_value(observations, internal_state)
        return values, next_internal_state

    def act(self, observations: ObservationInput, internal_state: Dict[str, Any] = None) -> Tuple[Actions, Dict[str, Any]]:
        actions, _, next_internal_state = self.action_value(observations, internal_state)
        return actions, next_internal_state

    def get_buffer_value_specs(self):
        internal_state_shape = self._model.internal_state_shape
        if internal_state_shape is None:
            return []
        else:
            return [
                ValueSpec(
                    ValueEnum.INTERNAL_STATE,
                    self._model.internal_state_shape,
                    storage_strategy=[ValueStorageStrategyEnum.LAST_PER_ENV, ValueStorageStrategyEnum.LAST_FROM_PREVIOUS_SEQUENCE],
                    default_value=self._model.initial_internal_state
                )
            ]

    def zero_grad(self):
        self._model_optimzier.zero_grad()

    def learning_step(self):
        self._model_optimzier.step()

    def train(self):
        super().train()
        self._model.train()

    def eval(self):
        super().eval()
        self._model.eval()

