from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple

import numpy as np

from myrl.spec.buffer import ValueSpec
from myrl.utils.batch import iterate_batches_indices
from myrl.utils.convertion import index_in_dict


class AbstractBufferData:
    @abstractmethod
    def iterate_batches(self,
                        batch_size: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        """
        Yields batches of size batch_size one by one.
        Batch contains 3 values:
        1) Batch dict of value names to array of shape [batch_size, seq_length, *]
        seq_length can be 1 in case buffer does not support sequences
        2) Batch dict of last values from previous sequences. Used for internal states of algorithms, such as RNN.
        Shape is [batch_size, 1, *].
        Empty dict in case of no such values
        3) Actual lengths of sequences. It is useful because sequences from first returned values are padded
        Shape is [batch_size]
        Can be None in case buffer does not support sequences
        """
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

    @abstractmethod
    def add_continuous(self,
                       key: str,
                       value: np.ndarray):
        """
        Used to put additional data in container, so it can be obtained in batches.
        Data can be calculated based on values from current container. (GAE for example)
        :param key: key of new value
        :param value: array of shape [n_envs, L, *],
            where L is total number of entries per each environment (sum of all lengths)
        """
        pass

    @property
    @abstractmethod
    def continuous_values_by_envs(self) -> Dict[str, np.ndarray]:
        """
        :return: Returns dict of values organized in shapes [n_envs, L, *].
            Useful for calculating running statistics, such as GAE
        """
        pass

    @property
    @abstractmethod
    def values(self) -> Dict[str, np.ndarray]:
        """
        Returns values flattened by env dimension
        :return: Dict of numpy arrays of shape [N, L, *]
        """
        pass

    @property
    @abstractmethod
    def last_values_from_prev_sequence(self) -> Dict[str, np.ndarray]:
        """
        Returns flattened last values from previous sequences
        :return: Numpy array of shape [N, 1, *]
        """
        pass

    @property
    @abstractmethod
    def lengths(self) -> np.ndarray:
        """
        Returns flattened lengths of sequences returned as values
        :return: Numpy array of shape [N] with lengths of sequences for each value entry
        """
        pass


class AbstractBufferSample:
    @property
    @abstractmethod
    def values(self):
        pass

    @property
    @abstractmethod
    def lengths(self):
        pass


class AbstractReplayBuffer(ABC):
    def __init__(self, capacity: int):
        assert capacity > 0
        self._capacity = capacity
        self._is_initialized = False

    @abstractmethod
    def init(self, values: Sequence[ValueSpec], n_envs: int):
        """
        Initializes the buffer given value specifications. Initialize buffer before using it
        :param values: Sequence of value specifications, which describe what values and how they should be stored
        :param n_envs: Number of parallel environment running.
            It is used to understand, how much space buffer need to store data.
            capacity number of values stored per each environment
        """
        pass

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    @abstractmethod
    def is_full(self) -> bool:
        """
        Indicates if buffer is full
        It means that if new value is added and buffer is full
         either error is thrown or new value overrites one of the old one
        """
        pass

    @abstractmethod
    def __len__(self):
        """Number of total entries in buffer"""
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def add(self, entry: Dict[str, np.ndarray]):
        """
        Adds values to buffer
        :param entry: Dict, which contains keys as value names to values itself
        :return:
        """
        pass

    def all_data(self) -> AbstractBufferData:
        """
        :return: Object, which contains all available data in buffer
        """
        pass

    @property
    @abstractmethod
    def last_values_per_env(self) -> Dict[str, np.ndarray]:
        pass

    def sample(self,
               n: int = 1,
               with_duplicates: bool = False) -> AbstractBufferSample:
        """
        Samples data from buffer
        :param n: number of entries to sample
        :param with_duplicates: is duplicates are allowed in sample
        :return: Object, which contains sampled data
        """
        pass
