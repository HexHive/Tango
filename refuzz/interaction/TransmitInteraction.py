from __future__ import annotations
from interaction import InteractionBase
from networkio   import ChannelBase
from typing      import ByteString

class TransmitInteraction(InteractionBase):
    def __init__(self, data: ByteString):
        self._data = data

    def perform(self, channel: ChannelBase):
        channel.send(self._data)
        # TODO hook target's recv calls and identify packet boundaries

    def __eq__(self, other: TransmitInteraction):
        return self._data == other._data

    def mutate(self, mutator):
        self._data = bytearray(self._data)
        mutator.mutate_buffer(self._data)