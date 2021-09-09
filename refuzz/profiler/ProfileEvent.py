from profiler import ProfilerBase
from threading import Condition

class ProfileEvent(ProfilerBase):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._cv = Condition()
        self._args = None

    def __call__(self, obj):
        def func(*args, **kwargs):
            with self._cv:
                ret = obj(*args, **kwargs)
                self._args = (args, kwargs)
                self._ret = ret
                self._cv.notify_all()
        return func

    @property
    def value(self):
        raise NotImplemented()

    @property
    def args(self):
        return self._args

    @property
    def ret(self):
        return self._ret

    def __enter__(self):
        self._cv.__enter__()
        while self._args is None:
            self._cv.wait()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._args = None
        self._ret = None
        self._cv.__exit__(exc_type, exc_value, exc_traceback)