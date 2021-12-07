from abc import abstractmethod
from profiler import ProfilerBase, ProfilingStoppedEvent as stopped
from threading import Thread

class PeriodicProfiler(ProfilerBase):
    def __init__(self, name, period=1, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._period = period
            self._thread = Thread(target=self._thread_worker)

    def __call__(self, obj):
        if not self._init_called:
            self._thread.start()
        return obj

    def _thread_worker(self):
        while not stopped.wait(self._period):
            self._task()

    @abstractmethod
    def _task(self):
        pass