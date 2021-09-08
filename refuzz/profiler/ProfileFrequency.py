from profiler import PeriodicProfiler
from collections import deque
import datetime

class ProfileFrequency(PeriodicProfiler):
    def __init__(self, name, samples=10, **kwargs):
        super().__init__(name, **kwargs)
        self._samples = deque(maxlen=samples)
        self._frequency = None

    def __call__(self, obj):
        obj = super().__call__(obj)
        def func(*args, **kwargs):
            self._samples.append(datetime.datetime.now())
            obj(*args, **kwargs)
        return func

    def _task(self):
        if len(self._samples) >= 1:
            duration = datetime.datetime.now() - self._samples[0]
            self._frequency = len(self._samples) / duration.total_seconds()
        else:
            self._frequency = 0

    @property
    def value(self):
        return self._frequency