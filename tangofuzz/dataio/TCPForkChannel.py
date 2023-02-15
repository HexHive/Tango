from . import debug, info

from dataio import (ChannelBase,
                      PtraceForkChannel,
                      TCPChannel, TCPChannelFactory)
from subprocess import Popen
from dataclasses import dataclass
from functools import cached_property
from ptrace.syscall import SYSCALL_REGISTER

@dataclass(kw_only=True)
class TCPForkChannelFactory(TCPChannelFactory):
    fork_before_accept: bool

    def create(self, pobj: Popen, netns: str, *args, **kwargs) -> ChannelBase:
        self._pobj = pobj
        self._netns = netns
        ch = self.forkchannel
        ch.connect((self.endpoint, self.port))
        return ch

    @cached_property
    def forkchannel(self):
        if self.fork_before_accept:
            return TCPForkBeforeAcceptChannel(pobj=self._pobj,
                          netns=self._netns,
                          endpoint=self.endpoint, port=self.port,
                          timescale=self.timescale,
                          connect_timeout=self.connect_timeout,
                          data_timeout=self.data_timeout)
        else:
            return TCPForkAfterListenChannel(pobj=self._pobj,
                          netns=self._netns,
                          endpoint=self.endpoint, port=self.port,
                          timescale=self.timescale,
                          connect_timeout=self.connect_timeout,
                          data_timeout=self.data_timeout)

class TCPForkAfterListenChannel(TCPChannel, PtraceForkChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cb_socket_listening(self, process, syscall):
        self._invoke_forkserver(process)

class TCPForkBeforeAcceptChannel(TCPChannel, PtraceForkChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cb_socket_listening(self, process, syscall):
        pass
        # self._invoke_forkserver(process)

    def cb_socket_accepting(self, process, syscall):
        # The syscall instruction cannot be "cancelled". Instead we replace it
        # by a call to an invalid syscall so the kernel returns immediately, and
        # the forkserver is later invoked properly. Otherwise, the forkerser
        # process itself will accept the next connect request.
        self._syscall_num = syscall.syscall
        process.setreg(SYSCALL_REGISTER, -1)
        # we also change the name so that it is ignored by any unintended
        # listener
        syscall.name = "FORKSERVER"
        syscall.syscall = -1
        self._invoke_forkserver(process)

    def _invoke_forkserver(self, process):
        address = process.getInstrPointer() - 2
        self._inject_forkserver(process, address)

    def _cleanup_forkserver(self, process):
        # WARN the use of SYSCALL_REGISTER sets orig_rax, which differs from
        # 'rax' in this context.
        with process.regsctx():
            process.setreg('rax', self._syscall_num)
            super()._cleanup_forkserver(process)
