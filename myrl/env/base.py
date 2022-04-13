from abc import ABC, abstractmethod

import numpy as np

from myrl.spec.env import ActionSpec, ObservationSpec


class Environment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self, reset_mask: np.array = None) -> np.array:
        assert reset_mask is None or (len(reset_mask.shape) == 0 and reset_mask.shape[0] == self.n_envs)
        """
        Resets environment(s)
        :param reset_mask: 1d array with mask, which environments must be reset.
                            Array length must be equal to environment number
        :return: Initial observations. Shape: [N, *OBS], where N is env number and OBS is shape of observation
        """

    @abstractmethod
    def step(self, action: np.array, indices: np.array = None) -> np.array:
        assert len(action.shape) == 2 and (indices is None or action.shape[0] == len(indices)) \
                and 0 < action.shape[0] <= self.n_envs
        """
        Take an action in environment
        :param action: 2d array of actions: [N, M], where N - number of environments, M - action dimensions.
                                            In case of discrete actions, they presented as index of action (starting from 0)
        :param indices: Indices of environment, for which step should be applied
        :return: Observations. Shape: [N, *OBS], where N is env number and OBS is shape of observation
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, **kwargs):
        """
        Renders the environment
        Params and returns depends on env type
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_envs(self) -> int:
        """
        :return: Number of parallel environments running
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def action_spec(self) -> ActionSpec:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_spec(self) -> ObservationSpec:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Closes the environments if necessary
        """
