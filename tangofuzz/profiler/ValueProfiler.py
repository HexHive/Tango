from profiler import AbstractProfiler

class ValueProfiler(AbstractProfiler):
    def __call__(self, obj):
        self._value = obj
        return obj

    @property
    def value(self):
        return self._value