import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

from common import ColoredLogger
from functools import partial
import logging
logging.getLogger().handlers.clear()
logging.setLoggerClass(ColoredLogger)
root_logger = ColoredLogger()

import common, fuzzer, generator, input, interaction, loader, mutator, \
       dataio, profiler, ptrace, statemanager, tests

# FIXME this is currently broken after switching to asyncio.Event
# import atexit
# import profiler
# def exit_handler():
#     profiler.ProfilingStoppedEvent.set()
# atexit.register(exit_handler)
