from abc          import ABC, abstractmethod
from input        import InputBase

class StateBase(ABC):
    def __init__(self):
        self._last_input = None
        self._out_edges = None
        self._in_edges = None
        self._pred = None

    @property
    def last_input(self):
        return self._last_input

    @last_input.setter
    def last_input(self, value):
        self._last_input = value

    @abstractmethod
    def __hash__(self):
        pass

    @property
    def out_edges(self):
        if self._out_edges:
            yield from self._out_edges(data=True)

    @out_edges.setter
    def out_edges(self, fn_edges):
        self._out_edges = fn_edges

    @property
    def in_edges(self):
        if self._in_edges:
            yield from self._in_edges(data=True)

    @in_edges.setter
    def in_edges(self, fn_edges):
        self._in_edges = fn_edges

    @property
    def predecessor_transition(self):
        return self._pred

    @predecessor_transition.setter
    def predecessor_transition(self, transition):
        self._pred = transition

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def get_escaper(self) -> InputBase:
        pass