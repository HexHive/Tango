import sys, logging
logger = logging.getLogger("networkio")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .Channel                    import (ChannelBase, ChannelFactoryBase,
                                        TCPChannelFactory, UDPChannelFactory,
                                        TCPChannel, UDPChannel)