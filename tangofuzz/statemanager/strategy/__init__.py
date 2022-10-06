import sys, logging
logger = logging.getLogger("statemanager.strategy")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .StrategyBase         import StrategyBase
from .RandomStrategy       import RandomStrategy
from .UniformStrategy      import UniformStrategy