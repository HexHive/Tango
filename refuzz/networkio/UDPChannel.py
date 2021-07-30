from typing import Callable
from networkio import ChannelBase, PtraceChannel, TransportChannelFactory
from subprocess import Popen
from dataclasses import dataclass

@dataclass
class UDPChannelFactory(TransportChannelFactory):
    def create(self, pobj: Popen) -> ChannelBase:
        return UDPChannel(pobj,
                          self.tx_callback, self.rx_callback,
                          self.endpoint,    self.port,
                          timescale=self.timescale)

class UDPChannel(PtraceChannel):
    def __init__(self, pobj: Popen, tx_callback: Callable, rx_callback: Callable,
                 endpoint: str, port: int, timescale: float):
        super().__init__(pobj, tx_callback, rx_callback, timescale)
        raise NotImplemented()