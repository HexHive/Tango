import sys, logging
logger = logging.getLogger("statemanager.coverage")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))