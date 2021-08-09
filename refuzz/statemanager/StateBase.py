from abc          import ABC, abstractmethod
from input        import InputBase

class StateBase(ABC):
    def __init__(self):
        self._last_input = None

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @property
    def last_input(self):
        return self._last_input

    @last_input.setter
    def last_input(self, value):
        self._last_input = value

    @abstractmethod
    def get_escaper(self) -> InputBase:
        pass