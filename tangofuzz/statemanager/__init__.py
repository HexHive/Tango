import sys, logging
logger = logging.getLogger("statemanager")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .StateBase                     import StateBase
from .StateMachine                  import StateMachine
from .strategy.StrategyBase         import StrategyBase
from .strategy.RandomStrategy       import RandomStrategy

from .StateTrackerBase              import StateTrackerBase
from .coverage.CoverageState        import CoverageState
from .coverage.CoverageReader       import CoverageReader
from .coverage.CoverageStateTracker import CoverageStateTracker
from .grammar.GrammarState          import GrammarState

from .StateManager                  import StateManager