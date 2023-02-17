from profiler import PeriodicProfiler, NumericalProfiler
from collections import deque
from statistics import mean

class LambdaMeanProfiler(PeriodicProfiler, NumericalProfiler):
    def __init__(self, *, samples=10, **kwargs):
        super().__init__(**kwargs)
        self._samples = deque(maxlen=samples)
        self._mean = None

    def __call__(self, obj):
        obj = super().__call__(obj)
        self._obj = obj
        return self._obj

    def do_task(self):
        self._samples.append(self._obj())
        self._mean = mean(self._samples)

    @property
    def numerical_value(self):
        return self._mean