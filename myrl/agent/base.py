from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

import torch as T
import torch.distributions as dist

from myrl.agent.action import Actions, ObservationInput
from myrl.spec.env import ActionSpec
from myrl.utils.convertion import np_to_tensor


class Agent(ABC):
    def __init__(self,
                 action_spec: ActionSpec):
        self._is_training = False
        self._action_spec = action_spec

    def random_action(self, n: int = 1) -> Actions:
        """
        :param n: Number of actions to return
        :return: actions object
        """
        assert n > 0
        if self._action_spec.is_continuous:
            low, high = np_to_tensor((self._action_spec.min, self._action_spec.max))
            low, high= low.expand(n, *self._action_spec.shape), high.expand(n, *self._action_spec.shape)
            distribution = dist.Uniform(low=T.tensor(low), high=T.tensor(high))
        else:
            uniform_probs = T.ones(n) / n
            distribution = dist.Categorical(probs=uniform_probs)
        return Actions(distribution)

    @abstractmethod
    def act(self,
            observations: ObservationInput,
            internal_state: Dict[str, Any] = None) -> Tuple[Actions, Dict[str, Any]]:
        """
        Acting according to the current policy
        :param observations: Array of shape: [N, *OBS], where N - number of observations, *OBS - dimensions of observations
        :param internal_state: Internal state of agent for specific observation. Useful for recurrent NNs
            Either None in case of agent has no internal state or dict of internal state keys to values of length N.
        :return: Tuple of actions and internal states (or None in case of no internal state).
        """
        assert len(observations.observations.shape) > 1 and \
               (internal_state is None or len(internal_state) == observations.shape[0])
        raise NotImplementedError

    def get_buffer_value_specs(self):
        """
        :return: List of value, to initialize memory buffer
        """
        return []

    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError

    @abstractmethod
    def learning_step(self):
        raise NotImplementedError

    @property
    def is_training(self):
        return self._is_training

    def train(self):
        self._is_training = True

    def eval(self):
        self._is_training = False
