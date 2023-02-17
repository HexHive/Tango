import sys, logging
logger = logging.getLogger("loader")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .Environment                import Environment
from .StateLoader            import AbstractStateLoader
from .BaseStateLoader import BaseStateLoader
from .ProcessLoader import ProcessLoader
from .replay.ReplayStateLoader   import ReplayStateLoader
from .replay.ReplayForkStateLoader   import ReplayForkStateLoader