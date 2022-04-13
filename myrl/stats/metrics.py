import enum


class MetricEnum(enum.Enum):
    POLICY_LOSS = enum.auto()
    VALUE_LOSS = enum.auto()
    CRITIC_LOSS = enum.auto()
    LOSS = enum.auto()
    ENTROPY_BONUS = enum.auto()
