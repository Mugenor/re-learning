from __future__ import annotations
from typing import Optional

import torch as T
from torch import distributions as dist

from myrl.agent.action import Actions
from myrl.spec.env import ActionSpec
from myrl.utils.convertion import unscale_continuous_action


class SACContinuousActions(Actions):
    def __init__(self,
                 distribution: dist.Distribution,
                 action_spec: ActionSpec):
        super().__init__(distribution=distribution, action=None)
        assert action_spec.is_continuous
        self._original_actions = None
        self._log_probs = None
        self._not_scaled_action = None
        self._action = None
        self._action_mean = T.tensor((action_spec.max + action_spec.min) / 2.0)
        self._action_half_spread = T.tensor((action_spec.max - action_spec.min) / 2.0)

    def action(self, reparam: bool = True, scale: bool = True) -> T.Tensor:
        if self._action is None:
            assert not reparam or self.distribution.has_rsample
            self._is_rsampled = reparam
            self._original_actions = self.distribution.rsample() if reparam else self.distribution.sample()
            self._not_scaled_action = T.tanh(self._original_actions)
            if scale:
                self._action = self._action_half_spread * self._not_scaled_action + self._action_mean
            else:
                self._action = self._not_scaled_action
        return self._action

    def log_prob(self, another_actions: Optional[T.Tensor] = None) -> T.Tensor:
        assert another_actions is None
        if self._log_probs is not None:
            return self._log_probs
        log_probs = self._distribution.log_prob(self._original_actions).sum(dim=-1, keepdims=True)
        log_probs -= T.sum(T.log(1 - self._not_scaled_action.pow(2) + 1e-6),
                           dim=-1,
                           keepdim=True)
        self._log_probs = log_probs
        return self._log_probs

    def unscale(self, action_spec: ActionSpec) -> Actions:
        """
        Transforms actions from [low:high] to [-1:1] and return new actions object.
        """
        assert action_spec.is_continuous
        self.action()
        new_action = Actions(action=self._not_scaled_action)
        new_action._copy_metadata_from(self)
        return new_action
