from profiler import ProfilerBase
import datetime
now = datetime.datetime.now

class ProfileTimeElapsed(ProfilerBase):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._now = now()

    def __call__(self, obj):
        raise NotImplemented()

    @property
    def value(self):
        time = now() - self._now
        return str(time)