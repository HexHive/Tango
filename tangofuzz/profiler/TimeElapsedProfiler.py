from profiler import AbstractProfiler
import datetime
now = datetime.datetime.now

class TimeElapsedProfiler(AbstractProfiler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._start = now()
        self._running = True
        self._accum = datetime.timedelta()

    def __call__(self, obj):
        raise NotImplementedError()

    def toggle(self):
        if self._running:
            self._accum += now() - self._start
            self._start = now()
            self._running = False
        else:
            self._start = now()
            self._running = True

    @property
    def value(self):
        time = now() - self._start + self._accum
        return str(time).split('.')[0]