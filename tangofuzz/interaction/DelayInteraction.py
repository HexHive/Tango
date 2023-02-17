from __future__ import annotations
from interaction import AbstractInteraction
from dataio   import AbstractChannel
from typing      import Sequence
from math        import isclose
from asyncio     import sleep

class DelayInteraction(AbstractInteraction):
    def __init__(self, time: float):
        self._time = float(time)
        # a delay longer than 5 seconds is probably useless, but we'll keep this
        # configurable
        self._maxdelay = 5

    async def perform(self, channel: AbstractChannel):
        await sleep(self._time * (await channel.timescale))

    def __eq__(self, other: DelayInteraction):
        return isinstance(other, DelayInteraction) and \
            isclose(self._time, other._time, rel_tol=0.01)

    def __repr__(self):
        return f'sleep({self._time})'

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__init__(self._time)
        return result