import sys, logging
logger = logging.getLogger("ptrace")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .signames import SIGNAMES, signalName   # noqa
from .errors import PtraceError   # noqa
