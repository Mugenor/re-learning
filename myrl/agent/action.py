from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch as T
import torch.nn.functional as F
from torch import distributions as dist

from myrl.spec.env import ActionSpec
from myrl.utils.convertion import np_to_tensor, unscale_continuous_action


@dataclass
class Actions:
    def __init__(self,
                 distribution: dist.Distribution = None,
                 action: T.Tensor = None):
        assert (distribution is not None) != (action is not None)
        self._distribution = distribution
        self._np_action = action
        self._action = action
        self._is_rsampled = None
        self._log_probs = None

    @property
    def distribution(self) -> Optional[dist.Distribution]:
        return self._distribution

    def action(self, reparam: bool = False, scale: bool = True) -> T.Tensor:
        """
        Lazily samples the action
        :param reparam: Should reparameterization trick be used or not during sampling
        :param scale: Should action be scaled or not. Ignored by default, but this parameter can be used in subclasses.
        :return:
        """
        if self._action is None:
            assert not reparam or self.distribution.has_rsample
            self._is_rsampled = reparam
            self._action = self.distribution.rsample() if reparam else self.distribution.sample()
        return self._action

    def np_action(self) -> np.ndarray:
        if self._np_action is None:
            self._np_action = self.action().detach().cpu().numpy()
        return self._np_action

    @property
    def n(self):
        if self._action is not None:
            return self._action.shape[0]
        else:
            return self._distribution.batch_shape[0]

    @property
    def is_rsampled(self):
        return self._is_rsampled

    def log_prob(self, another_actions: Optional[T.Tensor] = None):
        if self._distribution is None:
            raise RuntimeError("Distribution is not defined")
        if another_actions is None:
            if self._log_probs is None:
                self._log_probs = self._distribution.log_prob(self._action)
            return self._log_probs
        else:
            return self._distribution.log_prob(another_actions)

    def to_one_hot(self, num_classes: int = None) -> Actions:
        assert num_classes is not None or len(self._distribution.event_shape) > 0
        current_action = self.action()
        if num_classes is not None:
            if len(current_action.size()) > 1 and current_action.size()[-1] == num_classes:
                return self
            one_hot_action = F.one_hot(current_action.long(),
                                       num_classes=num_classes)
        else:
            if current_action.size() != self._distribution.batch_shape:
                # it is already one hot encoded
                return self
            one_hot_action = F.one_hot(current_action.long(),
                                       num_classes=self._distribution.event_shape[0])
        new_action = Actions(action=one_hot_action)
        new_action._copy_metadata_from(self)
        return new_action

    def unscale(self, action_spec: ActionSpec) -> Actions:
        """
        In case of continuous actions transforms actions from [low:high] to [-1:1] and return new actions object.
        In other case returns self
        """
        if action_spec.is_continuous:
            current_action = self.action()
            unscaled_action = unscale_continuous_action(current_action, action_spec)
            new_action = Actions(action=unscaled_action)
            new_action._copy_metadata_from(self)
            return new_action
        else:
            return self

    def _copy_metadata_from(self, other: Actions):
        self._distribution = other._distribution
        self._is_rsampled = other._is_rsampled
        self._log_probs = other.log_prob()


@dataclass
class ObservationInput:
    def __init__(self,
                 observations: Union[T.Tensor, np.ndarray],
                 lengths: Optional[Union[T.Tensor, np.ndarray]] = None):
        """
        :param observations: Expected shape: [N, *OBS] or [N, L, *OBS] in case of sequences
        :param lengths: None in case observations is not sequences, or in other case tensor of shape [N]
        """
        assert lengths is None or lengths.shape[0] == observations.shape[0]
        self.observations = observations if isinstance(observations, T.Tensor) else np_to_tensor(observations)
        self.lengths = lengths if isinstance(lengths, T.Tensor) or lengths is None else np_to_tensor(lengths)

    @property
    def is_sequential(self) -> bool:
        return self.lengths is not None
