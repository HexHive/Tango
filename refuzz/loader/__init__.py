import sys, logging
logger = logging.getLogger("loader")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .Environment                import Environment
from .StateLoaderBase            import StateLoaderBase
from .replay.ReplayStateLoader   import ReplayStateLoader