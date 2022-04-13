import gym
import numpy as np
from gym import ObservationWrapper


class VelocityMaskWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env, velocity_mask: np.ndarray = np.array([1, 0, 1, 0], dtype=np.float32)):
        super().__init__(env)
        self._velocity_mask = velocity_mask

    def observation(self, observation):
        return observation * self._velocity_mask
