import sys, logging
logger = logging.getLogger("generator")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .InputGenerator import AbstractInputGenerator
from .BaseInputGenerator import BaseInputGenerator
from .RandomInputGenerator import RandomInputGenerator
from .ReactiveInputGenerator import ReactiveInputGenerator
from .StatelessReactiveInputGenerator import StatelessReactiveInputGenerator
