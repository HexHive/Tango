from __future__ import annotations
from interaction import InteractionBase
from networkio   import ChannelBase
from typing      import ByteString
from profiler    import ProfileFrequency, ProfileEvent

class ReceiveInteraction(InteractionBase):
    def __init__(self, size: int = 0, data: ByteString = None):
        self._size = size
        self._expected = data

    @ProfileEvent("perform_interaction")
    @ProfileFrequency("interactions", period=1)
    def perform(self, channel: ChannelBase):
        self.data = channel.receive()
        # TODO verify size? verify data??

    def __eq__(self, other: ReceiveInteraction):
        # FIXME make this depend on the expected data?
        return isinstance(other, ReceiveInteraction)

    def mutate(self, mutator, entropy):
        pass
