from typing import Dict

import numpy as np

from myrl.memory.base import AbstractBufferData
from myrl.utils.batch import iterate_batches_indices
from myrl.utils.convertion import split_to_sequences, concatenate_sequences, index_in_dict


class RecurrentBufferData(AbstractBufferData):
    def __init__(self,
                 values: Dict[str, np.ndarray],
                 last_values_from_prev_sequence: Dict[str, np.ndarray],
                 lengths: np.ndarray,
                 sequences_per_env: np.ndarray):
        # [n_envs, N, L, *]
        self._values = values
        # [n_envs, N, *]
        self._last_values_from_prev_sequence = last_values_from_prev_sequence
        # [n_envs, N]
        self._lengths = lengths
        # [n_envs]
        self._sequences_per_env = sequences_per_env
        self._continuous_values_by_envs = None
        self._flat_values = None
        self._flat_last_values_from_prev_sequence = None
        self._flat_lengths = None

    def add_continuous(self, key: str, value: np.ndarray):
        assert value.shape[0] == self._lengths.shape[0] and value.shape[1] == self._lengths.sum(axis=1)[0]
        max_seq_length = self._lengths.max()
        sequenced_value = np.zeros((self._lengths.shape[0], self._lengths.shape[1], max_seq_length, *value.shape[2:]),
                                   dtype=value.dtype)
        for env_index in range(value.shape[0]):
            sequenced_value[env_index] = split_to_sequences(value[env_index], self._lengths[env_index])
        self._values[key] = sequenced_value
        self._continuous_values_by_envs = None
        self._flat_values = None

    def iterate_batches(self, batch_size: int):
        flat_values = self.values
        flat_last_values_from_prev_sequence = self.last_values_from_prev_sequence
        flat_lengths = self.lengths
        for indices in iterate_batches_indices(len(flat_lengths), batch_size):
            lengths_batch = flat_lengths[indices]
            max_length_per_batch = lengths_batch.max()
            values_slice = (indices, slice(max_length_per_batch))
            values_batch = index_in_dict(flat_values, values_slice)
            last_values_from_prev_sequence_batch = index_in_dict(flat_last_values_from_prev_sequence, values_slice)
            yield values_batch, last_values_from_prev_sequence_batch, lengths_batch

    @property
    def continuous_values_by_envs(self) -> Dict[str, np.ndarray]:
        if self._continuous_values_by_envs is not None:
            return self._continuous_values_by_envs
        lengths_by_env_index = self._lengths.sum(axis=1)
        assert (lengths_by_env_index == lengths_by_env_index[0]).all(), \
            "It is expected, the total sequence length is the same for each env"
        sequence_length = lengths_by_env_index[0]
        continuous_values = {key: np.zeros((value.shape[0], sequence_length, *value.shape[3:]), dtype=value.dtype)
                             for key, value in self._values.items()}
        for env_index in range(self._lengths.shape[0]):
            env_lengths = self._lengths[env_index]
            env_seq_count = self._sequences_per_env[env_index]
            concatenated_sequences = concatenate_sequences(
                [value[env_index, :env_seq_count] for value in self._values.values()],
                env_lengths[:env_seq_count]
            )
            for key, concatenated_seq in zip(self._values.keys(), concatenated_sequences):
                continuous_values[key][env_index] = concatenated_seq
        self._continuous_values_by_envs = continuous_values
        return continuous_values

    @property
    def values(self) -> Dict[str, np.ndarray]:
        if self._flat_values is not None:
            return self._flat_values
        concatenated_values = concatenate_sequences(list(self._values.values()), self._sequences_per_env)
        self._flat_values = {key: array
                             for key, array in zip(self._values.keys(), concatenated_values)}
        return self._flat_values

    @property
    def last_values_from_prev_sequence(self) -> Dict[str, np.ndarray]:
        if self._flat_last_values_from_prev_sequence is not None:
            return self._flat_last_values_from_prev_sequence
        concatenated_values = concatenate_sequences(
            list(self._last_values_from_prev_sequence.values()),
            self._sequences_per_env
        )
        self._flat_last_values_from_prev_sequence = {key: array
                                                     for key, array in
                                                     zip(self._last_values_from_prev_sequence.keys(), concatenated_values)}
        return self._flat_last_values_from_prev_sequence

    @property
    def lengths(self) -> np.ndarray:
        if self._flat_lengths is not None:
            return self._flat_lengths
        self._flat_lengths = concatenate_sequences(self._lengths, self._sequences_per_env)
        return self._flat_lengths


class RecurrentBufferSample:
    def __init__(self,
                 values: Dict[str, np.ndarray],
                 lengths: np.ndarray,
                 underlying_indices: np.ndarray):
        # [N, L, *OBS]
        self._values = values
        self._lengths = lengths
        self._underlying_indices = underlying_indices

    def iterate_batches(self, batch_size: int):
        for indices in iterate_batches_indices(len(self._lengths), batch_size):
            values_batch = {}
            for key, value in self._values:
                values_batch[key] = value[indices]
            yield values_batch, self._lengths[indices]

    @property
    def values(self) -> Dict[str, np.ndarray]:
        return self._values

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    @property
    def underlying_indices(self) -> np.ndarray:
        return self._underlying_indices
