from abc import ABC, abstractmethod
from collections.abc import Iterable
from tracker import AbstractState
from input import AbstractInput
from profiler import FrequencyProfiler
from random import Random

class AbstractInputGenerator(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        setattr(cls, 'generate', FrequencyProfiler('gens')(cls.generate))

    @abstractmethod
    def update_state(self, state: AbstractState, input: AbstractInput, *, exc: Exception=None, **kwargs):
        pass

    @abstractmethod
    def update_transition(self, source: AbstractState, destination: AbstractState, input: AbstractInput, *, state_changed: bool, exc: Exception=None, **kwargs):
        pass

    @abstractmethod
    def generate(self, state: AbstractState, entropy: Random) -> AbstractInput:
        pass

    @abstractmethod
    @property
    def seeds(self) -> Iterable[AbstractInput]:
        pass

    @abstractmethod
    @property
    def startup_input(self) -> AbstractInput:
        pass