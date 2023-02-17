import sys, logging
logger = logging.getLogger("mutator")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .BaseMutator import BaseMutator
from .HavocMutator import HavocMutator
from .ReactiveHavocMutator import ReactiveHavocMutator