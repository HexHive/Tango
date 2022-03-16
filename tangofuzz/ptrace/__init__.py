import sys, logging
logger = logging.getLogger("ptrace")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from ptrace.signames import SIGNAMES, signalName   # noqa
from ptrace.errors import PtraceError   # noqa
from ptrace.version import VERSION, __version__  # noqa
