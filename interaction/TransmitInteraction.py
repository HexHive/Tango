from interaction import InteractionBase
from loader      import ChannelBase
from typing      import ByteString, Sequence

class TransmitInteraction(InteractionBase):
    def __init__(self, data: ByteString):
        self._data = data

    def perform(self, channel: ChannelBase) -> Sequence[InteractionBase]:
        channel.send(self._data)
        # TODO hook target's recv calls and identify packet boundaries

        return [self,]
