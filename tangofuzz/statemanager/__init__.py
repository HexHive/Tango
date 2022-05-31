import sys, logging
logger = logging.getLogger("statemanager")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .StateBase                     import StateBase
from .StateMachine                  import StateMachine
from .strategy.StrategyBase         import StrategyBase
from .strategy.RandomStrategy       import RandomStrategy
from .strategy.UniformStrategy      import UniformStrategy
from .strategy.ZoomStrategy         import ZoomStrategy

from .StateTrackerBase              import StateTrackerBase
from .coverage.CoverageState        import CoverageState
from .coverage.CoverageReader       import CoverageReader
from .coverage.CoverageStateTracker import CoverageStateTracker
from .grammar.GrammarState          import GrammarState
from .zoom.ZoomStateReader          import ZoomStateReader, ZoomFeedback
from .zoom.ZoomState                import ZoomState
from .zoom.ZoomStateTracker         import ZoomStateTracker

from .StateManager                  import StateManager