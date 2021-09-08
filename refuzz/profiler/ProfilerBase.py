from abc import ABC, abstractmethod
from profiler import ProfiledObjects as objects

class ProfilerBase(ABC):
    def __init__(self, name):
        self._name = name
        objects[self._name] = self

    @abstractmethod
    def __call__(self, obj):
        pass

    @property
    @abstractmethod
    def value(self):
        pass