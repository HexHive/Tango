import sys, logging
logger = logging.getLogger("interaction")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .InteractionBase     import InteractionBase
from .DelayInteraction    import DelayInteraction
from .ReceiveInteraction  import ReceiveInteraction
from .TransmitInteraction import TransmitInteraction
from .MoveInteraction     import MoveInteraction
from .ShootInteraction    import ShootInteraction
from .ActivateInteraction import ActivateInteraction
from .ResetKeysInteraction import ResetKeysInteraction

from .RotateInteraction import RotateInteraction
from .ReachInteraction  import ReachInteraction
from .KillInteraction   import KillInteraction