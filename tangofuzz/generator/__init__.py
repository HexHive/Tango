import sys, logging
logger = logging.getLogger("generator")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .InputGeneratorBase import InputGeneratorBase
from .RandomInputGenerator import RandomInputGenerator
from .ZoomInputGenerator import ZoomInputGenerator