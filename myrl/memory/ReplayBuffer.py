from typing import Sequence, Tuple, Dict

import numpy as np

from myrl.memory.base import AbstractReplayBuffer
from myrl.memory.sample import BufferData, BufferSample
from myrl.spec.buffer import ValueSpec, ValueEnum, ValueStorageStrategyEnum


class ReplayBuffer(AbstractReplayBuffer):
    def __init__(self, capacity: int, is_circular: bool = True):
        """
        Simple buffer
        :param capacity: max elements in buffer
        :param is_circular: is buffer circular, i.e. starting overwriting old values when capacity is reached.
            In other case throws an error, if there is attempt to write in full buffer
        """
        super().__init__(capacity)
        self._capacity = capacity
        self._is_circular = is_circular
        # points to current cell
        self._pointer = 0
        self._all_env_indices = None
        # counts, how many times we added values
        self._counter = 0
        self._values = {}
        self._last_values_per_env = {}
        self._n_envs = -1
        self._is_initialized = False

    def init(self, values: Sequence[ValueSpec], n_envs: int):
        assert not self._is_initialized and n_envs > 0
        self._n_envs = n_envs
        for value_spec in values:
            name = value_spec.name_str
            if ValueStorageStrategyEnum.LAST_PER_ENV in value_spec.storage_strategy:
                arr = np.zeros((n_envs, *value_spec.shape), dtype=value_spec.dtype)
                self._last_values_per_env[name] = arr
            if ValueStorageStrategyEnum.EACH in value_spec.storage_strategy:
                arr = np.zeros((n_envs, self._capacity, *value_spec.shape), dtype=value_spec.dtype)
                self._values[name] = arr
        self.clear()
        self._is_initialized = True

    @property
    def is_full(self):
        return self._pointer + 1 >= self._capacity

    def __len__(self):
        return min(self._counter, self._capacity) * self._n_envs

    def clear(self):
        self._pointer = 0
        self._counter = 0
        self._all_env_indices = np.arange(self._n_envs)

    def add(self, entry: Dict[str, np.ndarray]):
        for key, array in self._last_values_per_env.items():
            entry_array = entry.get(key, None)
            if entry_array is None:
                raise RuntimeError(f"'{key}' key is not found in entry")
            array[self._all_env_indices] = entry_array
        for key, array in self._values.items():
            entry_array = entry.get(key, None)
            if entry_array is None:
                raise RuntimeError(f"'{key}' key is not found in entry")
            array[self._all_env_indices, self._pointer] = entry_array
        self._pointer = self._pointer + 1
        if self._is_circular:
            self._pointer %= self._capacity
        self._counter += 1

    @property
    def all_data(self) -> BufferData:
        assert self._is_initialized
        data = {}
        sl = slice(0, min(self._counter, self._capacity))
        for key, array in self._values.items():
            data[key] = array[sl]
        return BufferData(data)

    @property
    def last_values_per_env(self) -> Dict[str, np.ndarray]:
        return dict(self._last_values_per_env)

    def sample(self, n: int = 1, with_duplicates: bool = False) -> BufferSample:
        assert self._is_initialized
        values_filled = min(self._counter, self._capacity)
        max_index = values_filled * self._n_envs
        indices = np.random.choice(max_index, size=n, replace=with_duplicates)
        data = {}
        for key, array in self._values.items():
            # TODO: https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html#numpy.ravel_multi_index
            data[key] = self._flatten_array(array, values_filled)[indices]
        return BufferSample(data)

    def _flatten_array(self, array: np.array, values_filled: int) -> np.array:
        array = array[:, :values_filled]
        # original shape is: [n_envs, N, *entry_shape]
        # resulting shape is: [n_envs * values_filled, *entry_shape]
        array = array.reshape((-1, *array.shape[2:]))
        return array
