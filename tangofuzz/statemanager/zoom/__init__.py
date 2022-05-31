import sys, logging
logger = logging.getLogger("statemanager.zoom")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))