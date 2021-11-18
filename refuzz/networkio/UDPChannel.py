from networkio import ChannelBase, PtraceChannel, TransportChannelFactory
from subprocess import Popen
from dataclasses import dataclass

@dataclass
class UDPChannelFactory(TransportChannelFactory):
    def create(self, pobj: Popen) -> ChannelBase:
        return UDPChannel(pobj,
                          self.endpoint, self.port,
                          timescale=self.timescale)

class UDPChannel(PtraceChannel):
    def __init__(self, pobj: Popen, endpoint: str, port: int, timescale: float):
        super().__init__(pobj, timescale)
        raise NotImplemented()