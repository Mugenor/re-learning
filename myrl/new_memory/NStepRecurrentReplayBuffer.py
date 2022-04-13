from typing import Sequence, Dict

import numpy as np

from myrl.spec.buffer import ValueSpec, ValueEnum, ValueStorageStrategyEnum

VALUE_STRATEGY_TO_STORAGE_CLASS = {

}

class NStepRecurrentReplayBuffer:
    def __init__(self,
                 capacity: int,
                 max_sequence_length: int = 8,
                 is_circular: bool = True):
        self._capacity = capacity
        self._max_sequence_length = max_sequence_length
        self._is_circular = is_circular
        self._n_envs = -1
        self._n_step_return = -1
        self._storages = {}
        self._is_initialized = False

    def init(self,
             values: Sequence[ValueSpec],
             n_envs: int,
             n_step_return: int = 0):
        assert not self._is_initialized and n_envs > 0 and n_step_return >= 0
        self._n_envs = n_envs
        self._n_step_return = n_step_return
        dones_are_present = False
        for value_spec in values:
            name = value_spec.name_str
            dones_are_present = dones_are_present or \
                                (value_spec.name_str == ValueEnum.DONE.name and
                                 ValueStorageStrategyEnum.EACH in value_spec.storage_strategy)
            assert len(value_spec.storage_strategy) > 0
            value_storages = []
            for storage_strategy in value_spec.storage_strategy:
                storage_cls = VALUE_STRATEGY_TO_STORAGE_CLASS.get(storage_strategy, None)
                assert storage_cls is not None, f'{storage_strategy} is not supported for NStepRecurrentReplayBuffer'
                storage = storage_cls(
                    n_envs,
                    self._capacity,
                    self._max_sequence_length,
                    value_spec.shape,
                    value_spec.dtype
                )
                value_storages.append(storage)
            self._storages[name] = value_storages
        assert dones_are_present, \
            "It's expected that done flags are present for recurrent buffer and storage strategy is EACH"
        self.clear()
        self._is_initialized = True

    def clear(self):
        self._sequence_pointers = np.zeros(self._n_envs, dtype=np.int64)
        self._sequence_entry_pointers = np.zeros(self._n_envs, dtype=np.int64)

    def add(self, entry: Dict[str, np.ndarray]):
        assert self._is_initialized
        for key, storages in self._storages.items():
            entry_array = entry.get(key, None)
            assert entry_array is not None
            for storage in storages:
                storage.put(
                    entry_array,
                    self._sequence_pointer,
                    self._sequence_entry_pointers,
                    entry
                )
