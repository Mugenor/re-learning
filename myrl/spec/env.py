from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np



@dataclass
class ActionSpec:
    n_actions: int
    min: Union[int, float, np.array]
    max: Union[int, float, np.array]
    is_continuous: bool

    @property
    def shape(self) -> Tuple[int]:
        return (self.n_actions,) if self.is_continuous else (1,)

    def to_one_hot(self, actions: np.array) -> np.array:
        """
        Convert discrete actions from label to one hot representation
        :param actions: Expected shape: [N, 1] or [N], where N is number of independent actions
        :return: Converted actions in shape: [N, n_actions]
        """
        assert not self.is_continuous
        return np.eye(self.n_actions)[actions.squeeze()]

    def to_labels(self, actions: np.array) -> np.array:
        """
        Convert discrete actions from one hot to label representation
        :param actions: Expected shape [N, n_actions], where N is number of independent actions
        :return: Converted actions in shape [N, 1]
        """
        assert not self.is_continuous
        return actions.argmax(axis=1).reshape(-1, 1)


@dataclass
class ObservationSpec:
    shape: Tuple[int]
    is_visual: bool
