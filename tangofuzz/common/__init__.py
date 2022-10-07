import sys, logging
logger = logging.getLogger("common")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))


from .Exceptions import (LoadedException,
                        StabilityException,
                        StatePrecisionException,
                        ChannelSetupException,
                        ChannelBrokenException,
                        ChannelTimeoutException,
                        ProcessCrashedException,
                        ProcessTerminatedException,
                        StateNotReproducibleException)
from .Logger import ColoredLogger
from .AsyncIO import (async_property,
                     async_cached_property,
                     async_enumerate,
                     # wrappers:
                     sync_to_async,
                     async_suspendable,
                     Suspendable,
                     # singleton
                     GLOBAL_ASYNC_EXECUTOR)
