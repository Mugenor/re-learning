from typing import Tuple, Any, Callable

import torch as T
from torch import nn
import torch.distributions as dist

from myrl.agent.action import ObservationInput, Actions
from myrl.agent.sac_action import SACContinuousActions
from myrl.model.base import Policy, QValueFunction
from myrl.spec.env import ObservationSpec, ActionSpec
from myrl.utils.distribution import GumbelSoftmax


class SACGumbelSoftmaxPolicy(Policy):
    def __init__(self,
                 observation_spec: ObservationSpec,
                 action_spec: ActionSpec,
                 hidden=64):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Linear(observation_spec.shape[0], hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, action_spec.shape[0] if action_spec.is_continuous else action_spec.n_actions),
        )

    def forward(self, observations: ObservationInput, internal_state) -> Tuple[Actions, Any]:
        logits = self.pipe(observations.observations)
        return Actions(GumbelSoftmax(1.0, logits=logits)), None


def _weights_init(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias, 0)


class SACGaussianPolicy(Policy):
    def __init__(self,
                 observation_spec: ObservationSpec,
                 action_spec: ActionSpec,
                 hidden1=64, hidden2=64,
                 log_std_min=-20, log_std_max=2,
                 activation_fn_creator: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Linear(observation_spec.shape[0], hidden1),
            activation_fn_creator(),
            nn.Linear(hidden1, hidden2),
            activation_fn_creator()
        )
        self.mean_layer = nn.Linear(hidden2, action_spec.shape[0])
        self.std_layer = nn.Linear(hidden2, action_spec.shape[0])
        # self.apply(_weights_init)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self._action_spec = action_spec

    def forward(self,
                observations: ObservationInput,
                internal_state) -> Tuple[Actions, Any]:
        x = self.pipe(observations.observations)
        means = self.mean_layer(x)
        log_stds = self.std_layer(x)
        # Normalizing log std between min and max, so there should not be very small or large values for log_std
        # log_stds = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (T.tanh(log_stds) + 1)
        log_stds = T.clamp(log_stds, self.log_std_min, self.log_std_max)
        stds = log_stds.exp()
        return SACContinuousActions(
            distribution=dist.Normal(means, stds),
            action_spec=self._action_spec
        ), internal_state


class SACSimpleQValueFunction(QValueFunction):
    def __init__(self,
                 observation_spec: ObservationSpec,
                 action_spec: ActionSpec,
                 hidden=64,
                 activation_fn_creator: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Linear(observation_spec.shape[0] + action_spec.n_actions, hidden),
            activation_fn_creator(),
            nn.Linear(hidden, hidden),
            activation_fn_creator(),
            nn.Linear(hidden, 1)
        )

    def forward(self,
                observations: ObservationInput,
                actions: Actions, internal_state) -> Tuple[T.Tensor, Any]:
        return self.pipe(T.cat((observations.observations, actions.action()), dim=-1)), internal_state
