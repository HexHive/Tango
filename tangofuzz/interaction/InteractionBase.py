from __future__ import annotations
from abc           import ABC, abstractmethod
from networkio     import ChannelBase
from typing        import Sequence

class InteractionBase(ABC):
    @abstractmethod
    async def perform(self, channel: ChannelBase):
        pass

    @abstractmethod
    def mutate(self, mutator, entropy):
        pass

    @abstractmethod
    def __eq__(self, other: InteractionBase):
        pass