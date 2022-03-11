from . import debug, info

from networkio import (ChannelBase,
                      PtraceForkChannel,
                      TCPChannel,
                      TransportChannelFactory)
from subprocess import Popen
from dataclasses import dataclass
from functools import cached_property

@dataclass
class TCPForkChannelFactory(TransportChannelFactory):
    connect_timeout: float = None # seconds
    data_timeout: float = None # seconds

    protocol: str = "tcp"

    def create(self, pobj: Popen, netns: str) -> ChannelBase:
        self._pobj = pobj
        self._netns = netns
        ch = self.forkchannel
        ch.connect((self.endpoint, self.port))
        return ch

    @cached_property
    def forkchannel(self):
        return TCPForkChannel(pobj=self._pobj,
                          netns=self._netns,
                          endpoint=self.endpoint, port=self.port,
                          timescale=self.timescale,
                          connect_timeout=self.connect_timeout,
                          data_timeout=self.data_timeout)


class TCPForkChannel(TCPChannel, PtraceForkChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cb_socket_listening(self, process):
        self._invoke_forkserver(process)

    @property
    def forked_child(self):
        return self._accept_process
