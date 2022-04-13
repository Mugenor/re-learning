from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TypeVar, Generic

import torch as T

from myrl.model.base import RLModel
from myrl.utils.algorithm import hard_sync, soft_sync

RLModelType = TypeVar("RLModelType", bound=RLModel)


class TargetTwinModel(ABC, Generic[RLModelType]):
    def __init__(self, model: RLModelType):
        self._model = model
        self._target = deepcopy(model)
        self._target.eval()

    @property
    def model(self) -> RLModelType:
        return self._model

    @property
    def target(self) -> RLModelType:
        return self._target

    def parameters(self):
        return self._model.parameters()

    def named_parameters(self):
        return self._model.named_parameters()

    def to(self, device: T.device):
        self._model.to(device)
        self._target.to(device)
        return self

    def __repr__(self):
        return str(self._model)

    def train(self):
        self._model.train()
        return self

    def eval(self):
        self._target.eval()
        return self

    @abstractmethod
    def sync(self):
        pass


class HardTargetTwinModel(TargetTwinModel):
    def __init__(self, model: RLModel):
        super().__init__(model)

    def sync(self):
        hard_sync(self._model, self._target)


class SoftTargetTwinModel(TargetTwinModel):
    def __init__(self, model: RLModel, tau: float):
        super().__init__(model)
        self._tau = tau

    def sync(self):
        soft_sync(self._model, self._target, self._tau)
