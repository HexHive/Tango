from __future__ import annotations
from abc           import ABC, abstractmethod
from dataio     import AbstractChannel
from typing        import Sequence
from profiler      import FrequencyProfiler, EventProfiler

class AbstractInteraction(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        fn = EventProfiler('perform_interaction')(
            FrequencyProfiler('interactions', period=1)(cls.perform))
        setattr(cls, 'perform', fn)

    @abstractmethod
    async def perform(self, channel: AbstractChannel):
        pass

    @abstractmethod
    def __eq__(self, other: AbstractInteraction):
        pass