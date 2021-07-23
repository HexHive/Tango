import sys, logging
logger = logging.getLogger("input")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .InputBase     import InputBase
from .PreparedInput import PreparedInput
from .PCAPInput     import PCAPInput