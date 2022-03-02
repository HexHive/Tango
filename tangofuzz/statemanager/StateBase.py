from abc          import ABC, abstractmethod
from input        import InputBase

class StateBase(ABC):
    def __init__(self):
        self._last_input = None
        self._out_edges = None

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

    @property
    def out_edges(self):
        if self._out_edges:
            yield from self._out_edges(data=True)

    @out_edges.setter
    def out_edges(self, fn_edges):
        self._out_edges = fn_edges
