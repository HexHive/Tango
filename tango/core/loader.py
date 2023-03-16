from tango.core.tracker import AbstractState, AbstractTracker
from tango.core.types import Transition, LoadableTarget
from tango.common import AsyncComponent, ComponentType

from abc          import ABC, abstractmethod
from typing       import AsyncGenerator

__all__ = ['AbstractLoader', 'BaseLoader']

class AbstractLoader(AsyncComponent, ABC,
        component_type=ComponentType.loader):
    """
    The state loader maintains a running target and provides the capability of
    switching program states.
    """
    @abstractmethod
    async def load_state(self, state_or_path: LoadableTarget) \
            -> AsyncGenerator[Transition, AbstractState]:
        pass

    @abstractmethod
    async def apply_transition(self, transition: Transition,
            current_state: AbstractState) -> AbstractState:
        pass

class BaseLoader(AbstractLoader,
        capture_components={ComponentType.tracker}):
    def __init__(self, *, tracker: AbstractTracker, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker