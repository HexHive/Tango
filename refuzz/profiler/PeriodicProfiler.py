from abc import abstractmethod
from profiler import ProfilerBase, ProfilingStoppedEvent as stopped
from threading import Thread

class PeriodicProfiler(ProfilerBase):
    def __init__(self, name, period=1):
        super().__init__(name)
        self._period = period
        self._thread = Thread(target=self._thread)

    def __call__(self, obj):
        self._thread.start()
        return obj

    def _thread(self):
        while not stopped.wait(self._period):
            self._task()

    @abstractmethod
    def _task(self):
        pass