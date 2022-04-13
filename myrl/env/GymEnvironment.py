from typing import Dict, Tuple

import numpy as np
import gym

from myrl.env.base import Environment
from myrl.spec.env import ObservationSpec, ActionSpec


class GymEnvironment(Environment):
    def __init__(self, env_name: str, n_envs: int = 1, multiprocess=False, wrappers=None, seed=None):
        super().__init__()
        assert n_envs > 0
        self.env = gym.vector.make(env_name, num_envs=n_envs, asynchronous=multiprocess, wrappers = wrappers)
        self._action_spec = _action_space_to_spec(self.env.single_action_space)
        self._observation_spec = _obs_space_to_spec(self.env.single_observation_space)
        if seed is not None:
            self.env.seed(seed)
            self.env.single_action_space.seed(seed)
            self.env.single_observation_space.seed(seed)

    def reset(self, reset_mask: np.array = None) -> np.array:
        if reset_mask is None:
            return self.env.reset()
        else:
            observations = [env.reset() for env in np.array(self.env.envs)[reset_mask]]
            return np.array(observations)

    def step(self, action: np.array, indices=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        assert indices is None
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    @property
    def n_envs(self) -> int:
        return self.env.num_envs

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    @property
    def observation_spec(self) -> ObservationSpec:
        return self._observation_spec

    def close(self):
        self.env.close()


def _action_space_to_spec(action_space: gym.Space) -> ActionSpec:
    if isinstance(action_space, gym.spaces.Box):
        assert len(action_space.shape) == 1, "Multi dimensional spaces are not supported"
        return ActionSpec(
            n_actions=action_space.shape[0],
            min=action_space.low,
            max=action_space.high,
            is_continuous=True
        )
    elif isinstance(action_space, gym.spaces.Discrete):
        return ActionSpec(
            n_actions=action_space.n,
            min=0,
            max=action_space.n - 1,
            is_continuous=False
        )
    else:
        raise NotImplementedError(f'{action_space.__class__.__name__} is not supported as action space')


def _obs_space_to_spec(obs_space: gym.spaces.Space) -> ObservationSpec:
    if isinstance(obs_space, gym.spaces.Box):
        return ObservationSpec(
            shape=obs_space.shape,
            is_visual=False
        )
    elif isinstance(obs_space, gym.spaces.Discrete):
        return ObservationSpec(
            shape=(obs_space.n,),
            is_visual=False,
        )
    else:
        raise NotImplementedError(f'{obs_space.__class__.__name__} is not supported as action space')
