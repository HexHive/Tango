import sys, logging
logger = logging.getLogger("tracker")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .State                     import AbstractState
from .BaseState                 import BaseState
from .StateTracker              import AbstractStateTracker
from .LoaderDependentTracker    import LoaderDependentTracker