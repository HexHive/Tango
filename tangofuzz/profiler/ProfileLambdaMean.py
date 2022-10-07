from profiler import PeriodicProfiler
from collections import deque
from statistics import mean

class ProfileLambdaMean(PeriodicProfiler):
    def __init__(self, name, samples=10, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._samples = deque(maxlen=samples)
            self._mean = None

    def ___call___(self, obj):
        obj = super().___call___(obj)
        self._obj = obj
        return self._obj()

    def _task(self):
        self._samples.append(self._obj())
        self._mean = mean(self._samples)

    @property
    def value(self):
        return self.truncate(self._mean)