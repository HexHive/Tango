import sys, logging
logger = logging.getLogger("input")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .InputBase     import (InputBase, DecoratorBase,
                           JoiningDecorator, SlicingDecorator,
                           MemoryCachingDecorator)

from .PreparedInput import PreparedInput
from .SerializedInput import (SerializedInput, SerializedMetaInput,
                              FormatDescriptor, Serializer)
from .PCAPInput     import PCAPInput
from .RawInput     import RawInput
