import sys, logging
logger = logging.getLogger("input")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .InputBase     import (InputBase, DecoratorBase,
                           JoiningDecorator, SlicingDecorator,
                           MemoryCachingDecorator)
from .Decorators    import FileCachingDecorator
from .PreparedInput import PreparedInput
from .ZoomInput     import ZoomInput
from .PCAPInput     import PCAPInput