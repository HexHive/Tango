import sys, logging
logger = logging.getLogger("profiler")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

from threading import Event

ProfiledObjects = {}
ProfilingStoppedEvent = Event()

from sys import gettrace as sys_gettrace
DEBUG = sys_gettrace() is not None
ProfilingNOP = DEBUG

from .ProfilerBase import ProfilerBase
from .PeriodicProfiler import PeriodicProfiler
from .ProfileLambda import ProfileLambda
from .ProfileLambdaMean import ProfileLambdaMean
from .ProfileFrequency import ProfileFrequency
from .ProfileEvent import ProfileEvent
from .ProfileValue import ProfileValue
from .ProfileValueMean import ProfileValueMean
from .ProfileCount import ProfileCount
from .ProfileTimeElapsed import ProfileTimeElapsed