from __future__ import annotations
from interaction import InteractionBase
from networkio   import ChannelBase
from typing      import ByteString

class ReceiveInteraction(InteractionBase):
    def __init__(self, size: int = 0, data: ByteString = None):
        self._size = size
        self._expected = data

    def perform(self, channel: ChannelBase):
        self.data = channel.receive()
        # TODO verify size? verify data??

    def __eq__(self, other: ReceiveInteraction):
        # FIXME make this depend on the expected data?
        return True

    def mutate(self, mutator):
        pass