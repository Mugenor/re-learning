from typing import Tuple, Dict, Any

import numpy as np
import torch as T
import torch.nn as nn
import torch.distributions as dist

from myrl.agent.action import Actions, ObservationInput
from myrl.model.base import Policy, ValueFunction, PolicyValueSharedModel
from myrl.spec.env import ActionSpec, ObservationSpec


class PPOSimplePolicy(Policy):
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
            nn.Linear(hidden, action_spec.shape if action_spec.is_continuous else action_spec.n_actions),
            nn.Softmax(-1)
        )

    def forward(self, observations: ObservationInput, internal_state) -> Tuple[Actions, Any]:
        probs = self.pipe(observations.observations)
        return Actions(dist.Categorical(probs=probs)), None


class PPOSimpleValueFunction(ValueFunction):
    def __init__(self,
                 observation_spec: ObservationSpec,
                 hidded=64):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Linear(observation_spec.shape[0], hidded),
            nn.LeakyReLU(0.1),
            nn.Linear(hidded, hidded),
            nn.LeakyReLU(0.1),
            nn.Linear(hidded, 1)
        )

    def forward(self, observations: ObservationInput, internal_state) -> Tuple[T.Tensor, Any]:
        return self.pipe(observations.observations), None


class PPOSimpleRecurrentPolicyValueSharedModel(PolicyValueSharedModel):
    def __init__(self,
                 observation_spec: ObservationSpec,
                 action_spec: ActionSpec,
                 recurrent_hidden=64,
                 linear_hidden=128):
        super().__init__()
        self._observation_spec = observation_spec
        self.recurrent_layer = nn.LSTM(
            input_size=observation_spec.shape[0],
            hidden_size=recurrent_hidden,
            batch_first=True
        )
        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

        self.lin_hidden = nn.Linear(recurrent_hidden, linear_hidden)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        self.lin_policy = nn.Linear(linear_hidden, linear_hidden)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        self.lin_value = nn.Linear(linear_hidden, linear_hidden)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        self.policy = nn.Linear(linear_hidden, action_spec.shape if action_spec.is_continuous else action_spec.n_actions)
        nn.init.orthogonal_(self.policy.weight, np.sqrt(0.01))

        self.value = nn.Linear(linear_hidden, 1)
        nn.init.orthogonal_(self.value.weight, 1)

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, observations: ObservationInput, internal_state) -> Tuple[Actions, T.Tensor, Any]:
        obs_was_unsqueezed = False
        if internal_state is not None:
            internal_state = internal_state.transpose(0, 1)
            internal_state = T.chunk(internal_state, 2, dim=-1)
        if observations.is_sequential:
            x = nn.utils.rnn.pack_padded_sequence(
                observations.observations,
                observations.lengths,
                batch_first=True,
                enforce_sorted=False
            )
        else:
            x = observations.observations
            if len(self._observation_spec.shape) + 2 > len(x.shape):
                x = T.unsqueeze(x, 1)
                obs_was_unsqueezed = True
        x, new_internal_state = self.recurrent_layer(x, internal_state)
        # LSTM returns tuple of 2 tensors as internal state. Also swap batch and direction dimensions
        new_internal_state = T.cat(new_internal_state, dim=-1).transpose(0, 1)
        if observations.is_sequential:
            x, _ = nn.utils.rnn.pad_packed_sequence(
                x,
                batch_first=True,
            )
        elif obs_was_unsqueezed:
            x = T.squeeze(x, 1)
        x = self.activation(self.lin_hidden(x))
        x_policy = self.activation(self.lin_policy(x))
        x_value = self.activation(self.lin_value(x))
        action_logits = self.policy(x_policy)
        values = self.value(x_value)
        action_distribution = dist.Categorical(logits=action_logits)

        return Actions(action_distribution), values, new_internal_state

    @property
    def internal_state_shape(self):
        D = 2 if self.recurrent_layer.bidirectional else 1
        return (D, 2 * self.recurrent_layer.hidden_size)

    @property
    def initial_internal_state(self):
        return np.zeros(self.internal_state_shape)






