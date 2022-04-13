from typing import Dict, Tuple

import numpy as np

from myrl.memory.base import AbstractBufferData, AbstractBufferSample
from myrl.utils.algorithm import EMPTY_DICT


class BufferData(AbstractBufferData):
    def __init__(self, values: Dict[str, np.ndarray]):
        assert len(values) > 0
        # [n_envs, N, *]
        first_value = values[next(iter(values))]
        self._n_envs = first_value.shape[0]
        self._n_steps = first_value.shape[1]
        self._total_values = first_value.shape[0] * first_value.shape[1]
        self._values = values
        self._flat_values = None
        self._lengths = None

    def add_continuous(self, key: str, value: np.ndarray):
        assert value.shape[0] == self._n_envs and value.shape[1]
        self._values[key] = value
        self._flat_values = None

    @property
    def continuous_values_by_envs(self) -> Dict[str, np.ndarray]:
        return self._values

    @property
    def values(self) -> Dict[str, np.ndarray]:
        if self._flat_values is not None:
            return self._flat_values
        self._flat_values = {key: array.reshape((self._total_values, 1, *array.shape[2:]))
                             for key, array in self._values.items()}
        return self._flat_values

    @property
    def last_values_from_prev_sequence(self) -> Dict[str, np.ndarray]:
        return EMPTY_DICT

    @property
    def lengths(self) -> np.ndarray:
        if self._lengths is not None:
            return self._lengths
        self._lengths = np.ones(self._total_values, dtype=np.int8)
        return self._lengths


class BufferSample(AbstractBufferSample):
    def __init__(self, values):
        assert len(values) > 0
        # [n_envs, N, *]
        first_value = values[next(iter(values))]
        self._values = values
        self._lengths = np.ones(first_value.shape[:2])

    @property
    def values(self):
        return dict(self._values)

    @property
    def lengths(self):
        return self._lengths



