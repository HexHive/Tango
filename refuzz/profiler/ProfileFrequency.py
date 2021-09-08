from profiler import PeriodicProfiler

class ProfileFrequency(PeriodicProfiler):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._counter = 0
        self._frequency = None

    def __call__(self, obj):
        obj = super().__call__(obj)
        def func(*args, **kwargs):
            self._counter += 1
            obj(*args, **kwargs)
        return func

    def _task(self):
        self._frequency = self._counter / self._period
        self._counter = 0

    @property
    def value(self):
        return self._frequency