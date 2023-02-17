from profiler import AbstractProfiler

class LambdaProfiler(AbstractProfiler):
    def __call__(self, obj):
        self._obj = obj
        return obj

    @property
    def value(self):
        return self._obj()