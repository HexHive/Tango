import sys, logging
logger = logging.getLogger("webui")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .WebLogHandler import WebLogHandler
from .WebDataLoader import WebDataLoader
from .WebRenderer import WebRenderer