import sys, logging
logger = logging.getLogger("input")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .Input import AbstractInput
from .BaseInput     import (BaseInput, BaseDecorator,
                           JoiningDecorator, SlicingDecorator,
                           MemoryCachingDecorator)

from .PreparedInput import PreparedInput
from .SerializedInput import SerializedInput, SerializedInputMeta, Serializer
from .PCAPInput     import PCAPInput
from .RawInput     import RawInput
