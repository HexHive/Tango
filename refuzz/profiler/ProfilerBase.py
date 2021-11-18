from abc import ABC, abstractmethod
from profiler import ProfiledObjects as objects
from math import trunc

class ProfilerBase(ABC):
    def __init__(self, name, digits=1):
        self._name = name
        self._step = 10 ** digits
        objects[self._name] = self

    @abstractmethod
    def __call__(self, obj):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    def truncate(self, value):
        return trunc(value * self._step) / self._step