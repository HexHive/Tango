import sys, logging
logger = logging.getLogger("statemanager.strategy")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .Strategy         import AbstractStrategy
from .BaseStrategy         import BaseStrategy
from .RandomStrategy       import RandomStrategy
from .UniformStrategy      import UniformStrategy