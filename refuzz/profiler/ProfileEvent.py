from profiler import ProfilerBase
from threading import Event

class ProfileEvent(ProfilerBase):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._event = Event()

    def __call__(self, obj):
        def func(*args, **kwargs):
            obj(*args, **kwargs)
            self._event.set()
        return func

    @property
    def value(self):
        return self

    def __enter__(self):
        self._event.wait()

    def __exit__(self, type, value, traceback):
        self._event.clear()