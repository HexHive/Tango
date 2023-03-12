import sys, logging
logger = logging.getLogger("core")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .profiler import *
from .dataio import *
from .input import *
from .driver import *
from .loader import *
from .tracker import *
from .types import *
from .explorer import *
from .mutator import *
from .generator import *
from .strategy import *
from .session import *
from .config import *

__all__ = (dataio.__all__ + driver.__all__ + loader.__all__ + tracker.__all__ +
    explorer.__all__ + input.__all__ + mutator.__all__ + generator.__all__ +
    strategy.__all__ + profiler.__all__ + config.__all__ + session.__all__ +
    types.__all__)