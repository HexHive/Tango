from abc import ABC, ABCMeta, abstractmethod
from profiler import ProfiledObjects as objects, ProfilingNOP

class AbstractProfilerMeta(ABCMeta):
    def __call__(cls, name, /, *args, **kwargs):
        if ProfilingNOP:
            return cls.nop
        if (p := objects.get(name)) is None:
            p = objects[name] = super().__call__(*args, **kwargs)
        return p

    @staticmethod
    def nop(obj):
        return obj

class AbstractProfiler(ABC, metaclass=AbstractProfilerMeta):
    @abstractmethod
    def __call__(self, obj):
        return obj

    @property
    @abstractmethod
    def value(self):
        pass