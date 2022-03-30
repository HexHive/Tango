import sys, logging
logger = logging.getLogger("statemanager")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .StateBase                     import StateBase
from .StateMachine                  import StateMachine
from .strategy.StrategyBase         import StrategyBase
from .strategy.RandomStrategy       import RandomStrategy
from .strategy.UniformStrategy      import UniformStrategy

from .StateTrackerBase              import StateTrackerBase
from .coverage.CoverageState        import CoverageState
from .coverage.CoverageReader       import CoverageReader
from .coverage.CoverageStateTracker import CoverageStateTracker
from .stackhash.StackState          import StackState
from .stackhash.StackHashReader     import StackHashReader
from .stackhash.StackStateTracker   import StackStateTracker
from .grammar.GrammarState          import GrammarState

from .StateManager                  import StateManager