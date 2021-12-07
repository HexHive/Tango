import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

from common import ColoredLogger
from functools import partial
import logging
logging.getLogger().handlers.clear()
logging.setLoggerClass(ColoredLogger)
root_logger = ColoredLogger()

import common, fuzzer, input, interaction, loader, mutator, statemanager