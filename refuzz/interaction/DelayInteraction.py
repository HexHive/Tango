from interaction import InteractionBase
from networkio   import ChannelBase
from typing      import Sequence
from time        import sleep

class DelayInteraction(InteractionBase):
    def __init__(self, time: float):
        self._time = time

    def perform(self, channel: ChannelBase) -> Sequence[InteractionBase]:
        sleep(self._time * channel._timescale)

        return [self,]
