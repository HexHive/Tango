import sys, logging
logger = logging.getLogger("mutator")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .MutatorBase import MutatorBase
from .HavocMutator import HavocMutator
from .ZoomMutator import ZoomMutator