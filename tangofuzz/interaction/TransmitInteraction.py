from __future__ import annotations
from interaction import InteractionBase
from dataio   import ChannelBase
from typing      import ByteString

class TransmitInteraction(InteractionBase):
    def __init__(self, data: ByteString):
        self._data = data

    async def perform_internal(self, channel: ChannelBase):
        sent = await channel.send(self._data)
        if sent < len(self._data):
            self._data = self._data[:sent]
        # TODO hook target's recv calls and identify packet boundaries

    def __eq__(self, other: TransmitInteraction):
        return isinstance(other, TransmitInteraction) and self._data == other._data

    def mutate(self, mutator, entropy):
        self._data = bytearray(self._data)
        mutator.mutate_buffer(self._data, entropy)
