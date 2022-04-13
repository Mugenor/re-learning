from typing import Sequence, Dict

import numpy as np

from myrl.spec.buffer import ValueSpec, ValueStorageStrategyEnum


class EachStorage:
    def __init__(self,
                 capacity: int,
                 n_envs: int,
                 data_shape: Sequence[int],
                 dtype: np.dtype):
        self._capacity = capacity
        self._n_envs = n_envs
        self._data_shape = data_shape
        self._data = np.zeros((n_envs, capacity, *data_shape), dtype=dtype)

    def put(self, entry_array: np.ndarray, pointer: int):
        assert entry_array.shape == (self._n_envs, *self._data_shape)
        self._data[:, pointer] = entry_array


VALUE_STRATEGY_TO_STORAGE_CLASS = {
    ValueStorageStrategyEnum.EACH: EachStorage
}


class NStepReplayBuffer:
    def __init__(self, capacity: int, is_circular: bool = True):
        self._capacity = capacity
        self._is_circular = is_circular
        self._is_full = False
        self._pointer = -1
        self._storages = {}
        self._n_envs = -1
        self._n_step_return = -1
        self._is_initialized = False

    def init(self,
             values: Sequence[ValueSpec],
             n_envs: int,
             n_step_return: int = 0):
        assert not self._is_initialized and n_envs > 0 and n_step_return >= 0
        self._n_envs = n_envs
        self._n_step_return = n_step_return
        for value_spec in values:
            name = value_spec.name_str
            assert len(value_spec.storage_strategy) > 0
            value_storages = []
            for storage_strategy in value_spec.storage_strategy:
                storage_cls = VALUE_STRATEGY_TO_STORAGE_CLASS.get(storage_strategy, None)
                assert storage_cls is not None, f'{storage_strategy} is not supported for NStepReplayBuffer'
                storage = storage_cls(self._capacity, n_envs, value_spec.shape, value_spec.dtype)
                value_storages.append(storage)
            self._storages[name] = value_storages
        self.clear()
        self._is_initialized = True

    def clear(self):
        self._pointer = 0
        self._is_full = False

    def add(self, entry: Dict[str, np.ndarray]):
        assert self._is_initialized
        for key, storages in self._storages.items():
            entry_array = entry.get(key, None)
            assert entry_array is not None
            for storage in storages:
                storage.put(entry_array, self._pointer)
        self._pointer += 1
        if not self._is_full and self._pointer >= self._capacity:
            self._is_full = True
        if self._is_circular:
            self._pointer %= self._capacity
