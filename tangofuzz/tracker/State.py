from __future__ import annotations
from abc import ABC, abstractmethod

class AbstractState(ABC):
    @property
    def state_manager(self):
        return self._sman

    @state_manager.setter
    def state_manager(self, sman):
        self._sman = sman

    @property
    @abstractmethod
    def last_input(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @property
    @abstractmethod
    def out_edges(self):
        pass

    @property
    @abstractmethod
    def in_edges(self):
        pass

    @property
    @abstractmethod
    def predecessor_transition(self):
        pass

    @property
    @abstractmethod
    def __eq__(self, other: AbstractState):
        pass