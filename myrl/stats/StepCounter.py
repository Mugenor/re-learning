import numpy as np


class StepCounter:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.steps = 0
        self.dones = np.zeros((n_envs,))
        self.learn_times = 0

    def inc_dones(self, done_mask: np.array):
        self.dones += done_mask.flatten()

    def inc_steps(self):
        self.steps += 1

    @property
    def total_steps(self):
        return self.steps * self.n_envs

    def inc_learn(self):
        self.learn_times += 1
