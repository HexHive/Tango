import sys, logging
logger = logging.getLogger("statemanager.coverage")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from CoverageState        import CoverageState
from CoverageReader       import CoverageReader
from GlobalCoverage       import GlobalCoverage
from CoverageStateTracker import CoverageStateTracker