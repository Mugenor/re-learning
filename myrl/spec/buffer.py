from dataclasses import dataclass
from typing import Tuple, Union, Sequence, Any
import enum

import numpy as np


class ValueEnum(enum.Enum):
    OBSERVATION = enum.auto()
    REWARD = enum.auto()
    ACTION = enum.auto()
    DONE = enum.auto()
    NEXT_OBSERVATION = enum.auto()
    INFO = enum.auto()
    LOG_PROB = enum.auto()
    VALUE = enum.auto()  # result of V function
    INTERNAL_STATE = enum.auto()
    VALUE_INTERNAL_STATE = enum.auto()
    POLICY_INTERNAL_STATE = enum.auto()
    ADVANTAGE = enum.auto()


class ValueStorageStrategyEnum(enum.Enum):
    EACH = enum.auto()
    LAST_FROM_PREVIOUS_SEQUENCE = enum.auto()
    LAST_PER_ENV = enum.auto()


@dataclass
class ValueSpec:
    name: Union[str, ValueEnum]
    shape: Tuple[int]
    dtype: Union[np.dtype, object] = np.float32
    # is value is mandatory for agent or algorithm to work
    mandatory: bool = True
    storage_strategy: Sequence[ValueStorageStrategyEnum] = (ValueStorageStrategyEnum.EACH,)
    # used with LAST_FROM_PREVIOUS_SEQUENCE for example,
    # in case of first sequence or if previous sequence ended because of episode ended
    default_value: Any = None

    def __post_init__(self):
        assert len(self.storage_strategy) > 0

    @property
    def name_str(self):
        return self.name if isinstance(self.name, str) else self.name.name
