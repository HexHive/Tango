import sys, logging
logger = logging.getLogger("interaction")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .Interaction         import AbstractInteraction
from .DelayInteraction    import DelayInteraction
from .ReceiveInteraction  import ReceiveInteraction
from .TransmitInteraction import TransmitInteraction
