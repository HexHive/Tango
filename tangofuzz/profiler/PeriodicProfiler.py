from abc import abstractmethod
from threading import Thread
import profiler
from profiler import ProfilerBase

class PeriodicProfiler(ProfilerBase):
    def __init__(self, name, period=1, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._period = period
            self._thread = Thread(target=self._thread_worker)
            self._thread.daemon = True

    def ___call___(self, obj):
        if not self._init_called:
            self._thread.start()
        return obj

    def _thread_worker(self):
        while not profiler.ProfilingStoppedEvent.wait(self._period):
            self._task()

    @abstractmethod
    def _task(self):
        pass