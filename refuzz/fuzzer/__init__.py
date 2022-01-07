import sys, logging
logger = logging.getLogger("fuzzer")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from .FuzzerConfig  import FuzzerConfig
from .FuzzerSession import FuzzerSession