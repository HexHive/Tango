from __future__ import annotations
from interaction import InteractionBase
from dataio   import ChannelBase
from typing      import ByteString

class ReceiveInteraction(InteractionBase):
    def __init__(self, size: int = 0, data: ByteString = None, expected: ByteString = None):
        self._size = size
        self._expected = expected
        self._data = data

    async def perform_internal(self, channel: ChannelBase):
        self._data = await channel.receive()
        # TODO verify size? verify data??

    def __repr__(self):
        return f'<rx data="{self._data}">'

    def __eq__(self, other: ReceiveInteraction):
        # FIXME make this depend on the expected data?
        return isinstance(other, ReceiveInteraction)
