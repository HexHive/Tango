import sys, logging
logger = logging.getLogger("networkio")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .ChannelBase             import (ChannelBase, ChannelFactoryBase,
                                     TransportChannelFactory)

from .PtraceChannel           import PtraceChannel
from .PtraceForkChannel       import PtraceForkChannel
from .TCPChannel              import TCPChannel, TCPChannelFactory
from .TCPForkChannel          import TCPForkChannel, TCPForkChannelFactory
from .UDPChannel              import UDPChannel, UDPChannelFactory