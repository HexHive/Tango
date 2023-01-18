from __future__ import annotations
from abc           import ABC, abstractmethod
from dataio     import ChannelBase
from typing        import Sequence
from profiler      import ProfileFrequency, ProfileEvent

class InteractionBase(ABC):
    @ProfileEvent("perform_interaction")
    @ProfileFrequency("interactions", period=1)
    async def perform(self, channel: ChannelBase, *args, **kwargs):
        return await self.perform_internal(channel, *args, **kwargs)

    @abstractmethod
    async def perform_internal(self, channel: ChannelBase):
        pass

    @abstractmethod
    def __eq__(self, other: InteractionBase):
        pass