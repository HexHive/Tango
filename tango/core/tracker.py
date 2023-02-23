from __future__ import annotations

from tango.core.input import AbstractInput
from tango.common import AsyncComponent, ComponentType

from abc          import ABC, abstractmethod

__all__ = [
    'AbstractState', 'BaseState', 'AbstractStateTracker',
    # 'LoaderDependentTracker'
]

class AbstractState(ABC):
    def __init__(self, *, tracker: AbstractStateTracker):
        self._tracker = tracker

    @property
    def tracker(self) -> AbstractStateTracker:
        return self._tracker

    @property
    @abstractmethod
    def last_input(self) -> AbstractInput:
        pass

    @property
    @abstractmethod
    def out_edges(self) -> Iterable[Transition]:
        pass

    @property
    @abstractmethod
    def in_edges(self) -> Iterable[Transition]:
        pass

    @property
    @abstractmethod
    def predecessor_transition(self) -> Transition:
        pass

    @abstractmethod
    def __eq__(self, other: AbstractState) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

from tango.core.types import Transition

class BaseState(AbstractState):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_input = None
        self._out_edges = None
        self._in_edges = None
        self._pred = None

    @property
    def last_input(self) -> AbstractInput:
        return self._last_input

    @last_input.setter
    def last_input(self, value: AbstractInput):
        self._last_input = value

    @property
    def out_edges(self) -> Iterable[Transition]:
        if self._out_edges:
            yield from self._out_edges(data=True)

    @out_edges.setter
    def out_edges(self, fn_edges: Callable[..., Iterable[Transition]]):
        self._out_edges = fn_edges

    @property
    def in_edges(self) -> Iterable[Transition]:
        if self._in_edges:
            yield from self._in_edges(data=True)

    @in_edges.setter
    def in_edges(self, fn_edges: Callable[..., Iterable[Transition]]):
        self._in_edges = fn_edges

    @property
    def predecessor_transition(self) -> Transition:
        return self._pred

    @predecessor_transition.setter
    def predecessor_transition(self, transition: Transition):
        self._pred = transition

class AbstractStateTracker(AsyncComponent, ABC,
        component_type=ComponentType.tracker):
    @property
    @abstractmethod
    def entry_state(self) -> AbstractState:
        """
        The state of the target when it is first launched (with no inputs sent)

        :returns:   The state object describing the entry state.
        :rtype:     AbstractState
        """
        pass

    @property
    @abstractmethod
    def current_state(self) -> AbstractState:
        pass

    @abstractmethod
    def update(self, source: AbstractState, input: AbstractInput) -> AbstractState:
        pass

    @abstractmethod
    def peek(self, default_source: AbstractState, expected_destination: AbstractState) -> AbstractState:
        pass

    @abstractmethod
    def reset_state(self, state: AbstractState):
        """
        Informs the state tracker that the loader has reset the target into a
        state.
        """
        pass