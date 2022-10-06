import sys, logging
logger = logging.getLogger("statemanager")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .StateMachine                  import StateMachine
from .StateManager                  import StateManager