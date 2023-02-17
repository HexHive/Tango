from profiler import PeriodicProfiler, NumericalProfiler
from inspect import iscoroutinefunction
from functools import wraps

class FrequencyProfiler(PeriodicProfiler, NumericalProfiler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._counter = 0
        self._frequency = None

    def __call__(self, obj):
        obj = super().__call__(obj)
        @wraps(obj)
        def func(*args, **kwargs):
            self._counter += 1
            return obj(*args, **kwargs)
        @wraps(obj)
        async def afunc(*args, **kwargs):
            self._counter += 1
            return await obj(*args, **kwargs)

        if iscoroutinefunction(obj):
            return afunc
        else:
            return func

    def do_task(self):
        self._frequency = self._counter / self._period
        self._counter = 0

    @property
    def numerical_value(self):
        return self._frequency