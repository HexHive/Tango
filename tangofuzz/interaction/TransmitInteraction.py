from __future__ import annotations
from interaction import AbstractInteraction
from dataio   import AbstractChannel
from typing      import ByteString
from profiler import CountProfiler

class TransmitInteraction(AbstractInteraction):
    def __init__(self, data: ByteString):
        self._data = data

    async def perform(self, channel: AbstractChannel):
        sent = await channel.send(self._data)
        if sent < len(self._data):
            self._data = self._data[:sent]
        CountProfiler('bytes_sent')(sent)
        # TODO hook target's recv calls and identify packet boundaries

    def __eq__(self, other: TransmitInteraction):
        return isinstance(other, TransmitInteraction) and self._data == other._data

    def __repr__(self):
        return f'<tx data="{self._data}">'

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__init__(self._data)
        return result