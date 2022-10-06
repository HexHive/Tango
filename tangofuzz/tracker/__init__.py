import sys, logging
logger = logging.getLogger("tracker")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .StateBase                     import StateBase
from .StateTrackerBase              import StateTrackerBase