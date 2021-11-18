from profiler import ProfilerBase, ProfilingStoppedEvent
from threading import Condition, Thread
from functools import partial

class ProfileEvent(ProfilerBase):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._cv = Condition()
        self._args = None
        self._listeners = []

    def __del__(self):
        with self._cv:
            self._cv.notify_all()
        for th in self._listeners:
            th.join()

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

    def _listener_internal(self, cb, period):
        def worker():
            while True:
                if period is None and ProfilingStoppedEvent.is_set():
                    break
                elif period is not None and ProfilingStoppedEvent.wait(timeout=period):
                    break
                with self:
                    cb(*self._args[0], **self._args[1], ret=self._ret)

        th = Thread(target=worker)
        self._listeners.append(th)
        th.start()
        return None

    def listener(self, period=None):
        return partial(self._listener_internal, period=period)