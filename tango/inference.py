from __future__ import annotations

from . import debug, info, warning

from tango.core import UniformStrategy, AbstractState
from tango.cov import LoaderDependentTracker, CoverageStateTracker

from enum import Enum, auto

__all__ = ['StateInferenceStrategy', 'StateInferenceTracker']

class InferenceMode(Enum):
    Discovery = auto()
    Diversification = auto()
    CrossPollination = auto()

class StateInferenceTracker(LoaderDependentTracker,
        capture_paths=['tracker.native_lib']):
    def __init__(self, *, native_lib: str=None, **kwargs):
        super().__init__(**kwargs)
        self._cov_tracker = CoverageStateTracker(native_lib=native_lib,
            loader=self._loader)
        self._mode = InferenceMode.Discovery

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['tracker'].get('type') == 'inference'

    async def initialize(self):
        await self._cov_tracker.initialize()
        await super().initialize()

    @property
    def state_graph(self):
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.state_graph
            case _:
                pass

    @property
    def entry_state(self) -> AbstractState:
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.entry_state
            case _:
                pass

    @property
    def current_state(self) -> AbstractState:
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.current_state
            case _:
                pass

    def peek(self, default_source: AbstractState, expected_destination: AbstractState) -> AbstractState:
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.peek(default_source, expected_destination)
            case _:
                pass

    def reset_state(self, state: AbstractState):
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.reset_state(state)
            case _:
                pass

    def update_state(self, source: AbstractState, /, *, input: AbstractInput,
            exc: Exception=None, peek_result: Optional[AbstractState]=None) \
            -> Optional[AbstractState]:
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.update_state(source, input=input,
                    exc=exc, peek_result=peek_result)
            case _:
                pass

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, *,
            state_changed: bool, exc: Exception=None):
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.update_transition(source, destination,
                    input, state_changed=state_changed, exc=exc)
            case _:
                pass

class StateInferenceStrategy(UniformStrategy,
        capture_components={'tracker'}):
    def __init__(self, *, tracker: StateInferenceTracker, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'