from . import debug, info

from dataio import (ChannelBase,
                      PtraceForkChannel,
                      UDPChannel, UDPChannelFactory)
from subprocess import Popen
from dataclasses import dataclass
from functools import cached_property
from ptrace.syscall import SYSCALL_REGISTER
import socket

@dataclass(kw_only=True)
class UDPForkChannelFactory(UDPChannelFactory):
    fork_before_bind: bool

    def create(self, pobj: Popen, netns: str, *args, **kwargs) -> ChannelBase:
        self._pobj = pobj
        self._netns = netns
        ch = self.forkchannel
        ch.connect((self.endpoint, self.port))
        return ch

    @cached_property
    def forkchannel(self):
        if self.fork_before_bind:
            return UDPForkBeforeBindChannel(pobj=self._pobj,
                              netns=self._netns,
                              endpoint=self.endpoint, port=self.port,
                              timescale=self.timescale,
                              connect_timeout=self.connect_timeout,
                              data_timeout=self.data_timeout)
        else:
            return UDPForkChannel(pobj=self._pobj,
                              netns=self._netns,
                              endpoint=self.endpoint, port=self.port,
                              timescale=self.timescale,
                              connect_timeout=self.connect_timeout,
                              data_timeout=self.data_timeout)

class UDPForkChannel(UDPChannel, PtraceForkChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cb_socket_bound(self, process, syscall):
        self._invoke_forkserver(process)

    def connect(self, address: tuple):
        if not self._sockconnected:
            super().connect(address)
        else:
            self._socket = self.nssocket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.connect(address)
            self._poll_sync()

class UDPForkBeforeBindChannel(UDPChannel, PtraceForkChannel):
    # Forking just before a bind will result in the same socket being bound in
    # multiple child processes (because they all copy a reference of the
    # parent's sockfd), and this is not allowed behavior by Linux's TCP/IP
    # stack. It will result in EINVAL (errno 22: Invalid argument)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cb_socket_binding(self, process, syscall):
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
