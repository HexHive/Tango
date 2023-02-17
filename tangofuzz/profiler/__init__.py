import sys, logging
logger = logging.getLogger("profiler")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from asyncio import Event, create_task

ProfiledObjects = {}
ProfilingTasks = []

async def initialize():
    global ProfilingStoppedEvent
    ProfilingStoppedEvent = Event()
    for idx, coro in enumerate(ProfilingTasks):
        ProfilingTasks[idx] = create_task(coro)

from sys import gettrace as sys_gettrace
DEBUG = sys_gettrace() is not None
ProfilingNOP = DEBUG

from .Profiler import AbstractProfiler
from .NumericalProfiler import NumericalProfiler
from .PeriodicProfiler import PeriodicProfiler
from .LambdaProfiler import LambdaProfiler
from .LambdaMeanProfiler import LambdaMeanProfiler
from .FrequencyProfiler import FrequencyProfiler
from .EventProfiler import EventProfiler
from .ValueProfiler import ValueProfiler
from .ValueMeanProfiler import ValueMeanProfiler
from .CountProfiler import CountProfiler
from .TimeElapsedProfiler import TimeElapsedProfiler