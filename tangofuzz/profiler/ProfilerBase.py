from abc import ABC, abstractmethod
from profiler import ProfiledObjects as objects, ProfilingNOP
from math import trunc

class ProfilerBase(ABC):
    def __new__(cls, name, **kwargs):
        if (other := objects.get(name)) is not None:
            other._init_called = True
            return other
        else:
            return super(ProfilerBase, cls).__new__(cls)

    def __init__(self, name, digits=1):
        if name in objects:
            return
        self._name = name
        self._step = 10 ** digits
        objects[self._name] = self
        self._init_called = False

    def __call__(self, obj):
        if ProfilingNOP:
            return obj
        else:
            return self.___call___(obj)

    @abstractmethod
    def ___call___(self, obj):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    def truncate(self, value):
        return trunc(value * self._step) / self._step
