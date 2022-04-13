from typing import Dict, Any, Tuple, List, Optional

import torch as T

from myrl.agent.action import ObservationInput, Actions
from myrl.agent.base import Agent
from myrl.model.base import Policy, QValueFunction, RepresentationModel
from myrl.model.optimizer import Optimizer
from myrl.model.twin import TargetTwinModel
from myrl.spec.buffer import ValueEnum
from myrl.spec.env import ActionSpec


class SACAgent(Agent):
    def __init__(self,
                 action_spec: ActionSpec,

                 policy: Policy,
                 policy_optimizer: Optimizer,

                 q_functions: List[TargetTwinModel[QValueFunction]],
                 q_optimizers: List[Optimizer],

                 representation_model: Optional[TargetTwinModel[RepresentationModel]] = None,
                 representation_optimizer: Optional[Optimizer] = None):
        super().__init__(action_spec)
        self._policy = policy
        self._policy_optimizer = policy_optimizer
        self._q_functions = q_functions
        self._q_optimizers = q_optimizers
        self._representation_model = representation_model
        self._representation_optimizer = representation_optimizer

    def representation(self,
                       observations: ObservationInput,
                       internal_state: Dict[str, Any] = None) -> Tuple[ObservationInput, Dict[str, Any]]:
        if self._representation_model is None:
            return observations, internal_state
        model_internal_state = internal_state.get(ValueEnum.INTERNAL_STATE.name, None)
        representation, next_internal_state = self._representation_model.model(observations, model_internal_state)
        if next_internal_state is None:
            return representation, {}
        else:
            return representation, {ValueEnum.INTERNAL_STATE.name: next_internal_state}

    def representation_target(self,
                              observations: ObservationInput,
                              internal_state: Dict[str, Any] = None) -> Tuple[ObservationInput, Dict[str, Any]]:
        if self._representation_model is None:
            return observations, internal_state
        model_internal_state = internal_state.get(ValueEnum.INTERNAL_STATE.name, None)
        representation, next_internal_state = self._representation_model.target(observations, model_internal_state)
        # ignore internal state of target representation model
        return representation, {}

    def policy(self, representations: ObservationInput) -> Actions:
        return self._policy(representations, None)

    def q_values(self,
                 representations: ObservationInput,
                 actions: Actions) -> T.Tensor:
        """
        :return: Tuple of q values of shape [q_functions_num, batch_size, 1] and internal state dict
        """
        q_values = []
        for q_function in self._q_functions:
            q_value, _ = q_function.model(representations, actions, None)
            q_values.append(q_value)
        q_values = T.stack(q_values, dim=0)
        return q_values

    def q_targets(self,
                  representations: ObservationInput,
                  actions: Actions) -> T.Tensor:
        """
        :return: Tuple of q values of shape [q_functions_num, batch_size, 1] and internal state dict
        """
        with T.no_grad():
            q_targets = []
            for q_function in self._q_functions:
                q_target, _ = q_function.target(representations, actions, None)
                q_targets.append(q_target)
            q_targets = T.stack(q_targets, dim=0)
            return q_targets

    def act(self,
            observations: ObservationInput,
            internal_state: Dict[str, Any] = None) -> Tuple[Actions, Dict[str, Any]]:
        representations, new_internal_state = self.representation(observations, internal_state)
        actions, _ = self.policy(representations)
        return actions, new_internal_state

    def sync_targets(self):
        for q_function in self._q_functions:
            q_function.sync()
        if self._representation_model is not None:
            self._representation_model.sync()


    def zero_grad(self):
        self._policy_optimizer.zero_grad()
        for q_optimizer in self._q_optimizers:
            q_optimizer.zero_grad()
        if self._representation_optimizer is not None:
            self._representation_optimizer.zero_grad()

    def learning_step(self):
        self._policy_optimizer.step()
        for q_optimizer in self._q_optimizers:
            q_optimizer.step()
        if self._representation_optimizer is not None:
            self._representation_optimizer.step()

    def train(self):
        super().train()
        self._policy.train()
        for q_function in self._q_functions:
            q_function.train()
        if self._representation_model is not None:
            self._representation_model.train()

    def eval(self):
        super().eval()
        self._policy.eval()
        for q_function in self._q_functions:
            q_function.eval()
        if self._representation_model is not None:
            self._representation_model.eval()
