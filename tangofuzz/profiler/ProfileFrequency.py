from profiler import PeriodicProfiler
from inspect import iscoroutinefunction

class ProfileFrequency(PeriodicProfiler):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._counter = 0
            self._frequency = None

    def __call__(self, obj):
        obj = super().__call__(obj)
        def func(*args, **kwargs):
            self._counter += 1
            return obj(*args, **kwargs)
        async def afunc(*args, **kwargs):
            self._counter += 1
            return await obj(*args, **kwargs)

        if iscoroutinefunction(obj):
            return afunc
        else:
            return func

    def _task(self):
        self._frequency = self._counter / self._period
        self._counter = 0

    @property
    def value(self):
        return self.truncate(self._frequency)