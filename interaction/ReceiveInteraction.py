from interaction import InteractionBase
from loader      import ChannelBase
from typing      import ByteString, Sequence

class ReceiveInteraction(InteractionBase):
    def __init__(self, size = 0: int, data = None: ByteString):
        self._size = size
        self._expected = data

    def perform(self, channel: ChannelBase) -> Sequence[InteractionBase]:
        self.data = channel.receive()
        # TODO verify size? verify data??

        return [self,]
