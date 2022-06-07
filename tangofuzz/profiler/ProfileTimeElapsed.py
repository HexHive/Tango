from profiler import ProfilerBase
import datetime
now = datetime.datetime.now

class ProfileTimeElapsed(ProfilerBase):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._start = now()
        self._running = True
        self._accum = datetime.timedelta()

    def __call__(self, obj):
        raise NotImplemented

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