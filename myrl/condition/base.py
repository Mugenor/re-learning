from abc import ABC, abstractmethod


class Condition(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> bool:
        raise NotImplementedError