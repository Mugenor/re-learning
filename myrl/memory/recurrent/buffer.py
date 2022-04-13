from typing import Sequence, Dict

import numpy as np

from myrl.memory.base import AbstractReplayBuffer
from myrl.memory.recurrent.sample import RecurrentBufferSample, RecurrentBufferData
from myrl.spec.buffer import ValueSpec, ValueEnum, ValueStorageStrategyEnum
from myrl.utils.convertion import create_sequence_mask


class RecurrentReplayBuffer(AbstractReplayBuffer):
    def __init__(self,
                 capacity: int,
                 max_sequence_length: int = 8,
                 is_circular: bool = True):
        """
        Buffer which support storing recurrent sequence
        :param is_circular: is buffer circular, i.e. starting overwriting old values when capacity is reached.
            In other case throws an error, if there is attempt to write in full buffer
        """
        super().__init__(capacity)
        self._capacity = capacity
        self._max_sequence_length = max_sequence_length
        self._is_circular = is_circular
        self._sequence_pointers = None
        self._sequence_entry_pointers = None
        self._sequence_lengths = None
        self._sequences_per_env = None
        self._values = {}
        self._last_values_per_env = {}
        self._last_values_from_prev_sequence = {}

        self._value_specs = {}
        self._n_envs = -1
        self._all_env_indices = None
        self._is_initialized = False

    def init(self, values: Sequence[ValueSpec], n_envs: int):
        assert not self._is_initialized and n_envs > 0
        self._n_envs = n_envs
        dones_are_present = False
        for value_spec in values:
            name = value_spec.name_str
            dones_are_present = dones_are_present or \
                                (value_spec.name_str == ValueEnum.DONE.name and
                                 ValueStorageStrategyEnum.EACH in value_spec.storage_strategy)
            if ValueStorageStrategyEnum.LAST_PER_ENV in value_spec.storage_strategy:
                arr = np.zeros((n_envs, *value_spec.shape), dtype=value_spec.dtype)
                self._last_values_per_env[name] = arr
            if ValueStorageStrategyEnum.LAST_FROM_PREVIOUS_SEQUENCE in value_spec.storage_strategy:
                assert isinstance(value_spec.default_value, np.ndarray) and \
                       value_spec.default_value.shape == value_spec.shape
                arr = np.zeros((n_envs, self._capacity, *value_spec.shape), dtype=value_spec.dtype)
                arr[:, 0] = value_spec.default_value
                self._last_values_from_prev_sequence[name] = arr
            if ValueStorageStrategyEnum.EACH in value_spec.storage_strategy:
                arr = np.zeros((n_envs, self._capacity, self._max_sequence_length, *value_spec.shape),
                               dtype=value_spec.dtype)
                self._values[name] = arr
            self._value_specs[name] = value_spec
        assert dones_are_present, \
            "It's expected that done flags are present for recurrent buffer and storage strategy is EACH"
        self.clear()
        self._is_initialized = True

    @property
    def is_full(self):
        return (self._sequences_per_env >= self._capacity).all()

    def clear(self):
        self._sequence_pointers = np.zeros(self._n_envs, dtype=np.int64)
        self._sequence_entry_pointers = np.zeros(self._n_envs, dtype=np.int64)
        self._sequence_lengths = np.zeros((self._n_envs, self._capacity), dtype=np.int64)
        self._sequences_per_env = np.zeros(self._n_envs, dtype=np.int64)
        self._all_env_indices = np.arange(self._n_envs)

    def add(self, entry):
        for key, array in self._last_values_per_env.items():
            entry_array = entry.get(key, None)
            if entry_array is None:
                raise RuntimeError(f"'{key}' key is not found in entry")
            array[self._all_env_indices] = entry_array
        for key, array in self._values.items():
            entry_array = entry.get(key, None)
            if entry_array is None:
                raise RuntimeError(f"'{key}' key is not found in entry")
            array[self._all_env_indices, self._sequence_pointers, self._sequence_entry_pointers] = entry_array
        self._sequence_lengths[self._all_env_indices, self._sequence_pointers] += 1
        # count sequences when first entry is added to sequence
        self._sequences_per_env[self._sequence_entry_pointers == 0] += 1
        self._sequences_per_env = self._sequences_per_env.clip(0, self._capacity)
        self._sequence_entry_pointers += 1
        encountered_sequence_capacity = (self._sequence_entry_pointers >= self._max_sequence_length) \
            .flatten()
        dones = entry[ValueEnum.DONE.name] \
            .astype(np.bool) \
            .flatten()
        should_start_new_sequence = np.logical_or(encountered_sequence_capacity, dones)
        self._sequence_entry_pointers[should_start_new_sequence] = 0
        self._sequence_pointers[should_start_new_sequence] += 1
        if self._is_circular:
            self._sequence_pointers %= self._capacity
        self._sequence_lengths[should_start_new_sequence,
                               self._sequence_pointers[should_start_new_sequence]] = 0
        if should_start_new_sequence.any():
            seq_pointers_encountered_capacity = self._sequence_pointers[encountered_sequence_capacity]
            seq_pointers_dones = self._sequence_pointers[dones]
            for key, array in self._last_values_from_prev_sequence.items():
                entry_array = entry.get(key, None)
                value_spec = self._value_specs[key]
                if entry_array is None:
                    raise RuntimeError(f"'{key}' key is not found in entry")
                array[encountered_sequence_capacity, seq_pointers_encountered_capacity] = \
                    entry_array[encountered_sequence_capacity]
                array[dones, seq_pointers_dones] = value_spec.default_value

    @property
    def all_data(self) -> RecurrentBufferData:
        assert self._is_initialized
        data = {}
        last_values_from_prev_sequence = {}
        max_sequences_per_env = self._sequences_per_env.max()
        lengths = self._sequence_lengths[:, :max_sequences_per_env].copy()
        is_not_filled_sequences = ~create_sequence_mask(self._sequences_per_env)
        lengths[is_not_filled_sequences] = 0
        for key, value in self._values.items():
            data[key] = value[:, :max_sequences_per_env]
        for key, value in self._last_values_from_prev_sequence.items():
            last_values_from_prev_sequence[key] = value[:, :max_sequences_per_env]
        return RecurrentBufferData(data, last_values_from_prev_sequence, lengths, self._sequences_per_env)

    @property
    def last_values_per_env(self) -> Dict[str, np.ndarray]:
        return dict(self._last_values_per_env)

    def sample(self,
               n: int = 1,
               with_duplicates: bool = False) -> RecurrentBufferSample:
        total_sequences = self._sequences_per_env.sum()
        assert self._is_initialized and total_sequences >= n
        sampled_indices = np.random.choice(total_sequences, size=n, replace=with_duplicates)
        if self.is_full:
            real_indices = np.arange(self._n_envs * self._max_sequence_length)
        else:
            real_indices_per_env = tuple(np.arange(sequences_num) + i * self._max_sequence_length
                                         for i, sequences_num in enumerate(self._sequences_per_env))
            real_indices = np.row_stack(real_indices_per_env)
        lengths = self._sequence_lengths.reshape(-1)[real_indices[sampled_indices]]
        sample = {}
        for key, sequences_per_env in self._values.items():
            flat_sequences = sequences_per_env.reshape((-1, *sequences_per_env.shape[2:]))
            sample[key] = flat_sequences[real_indices[sampled_indices]]
        return RecurrentBufferSample(sample, lengths, real_indices)
