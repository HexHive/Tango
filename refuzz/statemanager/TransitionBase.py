from abc   import ABC, abstractmethod
from input import InputBase

class TransitionBase(ABC):
    @abstractmethod
    def __add__(self, other):
        pass

    @property
    @abstractmethod
    def input(self) -> InputBase:
        pass