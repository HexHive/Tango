import sys, logging
logger = logging.getLogger("interaction")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .InteractionBase     import InteractionBase
from .DelayInteraction    import DelayInteraction
from .ReceiveInteraction  import ReceiveInteraction
from .TransmitInteraction import TransmitInteraction