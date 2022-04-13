from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

import torch as T
import torch.nn as nn

from myrl.agent.action import Actions, ObservationInput


class RLModel(nn.Module, ABC):
    @property
    def internal_state_shape(self):
        return None

    @property
    def initial_internal_state(self):
        return None


class Policy(RLModel, ABC):
    @abstractmethod
    def forward(self, observations: ObservationInput, internal_state) -> Tuple[Actions, Any]:
        pass


class ValueFunction(RLModel, ABC):
    @abstractmethod
    def forward(self, observations: ObservationInput, internal_state) -> Tuple[T.Tensor, Any]:
        pass


class QValueFunction(RLModel, ABC):
    @abstractmethod
    def forward(self, observations: ObservationInput, actions: Actions, internal_state) -> Tuple[T.Tensor, Any]:
        pass


class QValueForEachActionFunction(RLModel, ABC):
    @abstractmethod
    def forward(self, observations: ObservationInput, internal_state) -> Tuple[T.Tensor, Any]:
        pass


class PolicyValueSharedModel(RLModel, ABC):
    @abstractmethod
    def forward(self, observations: ObservationInput, internal_state) -> Tuple[Actions, T.Tensor, Any]:
        pass


class RepresentationModel(RLModel, ABC):
    @abstractmethod
    def forward(self, observations: ObservationInput, internal_state) -> Tuple[ObservationInput, Any]:
        pass
