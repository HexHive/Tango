from abc import abstractmethod
from profiler import AbstractProfiler
from math import trunc

class NumericalProfiler(AbstractProfiler):
    def __init__(self, *, decimal_digits=1, **kwargs):
        super().__init__(**kwargs)
        self._step = 10 ** decimal_digits

    def truncate(self, value):
        return trunc(value * self._step) / self._step

    @property
    @abstractmethod
    def numerical_value(self):
        pass

    @property
    def value(self):
        return self.truncate(self.numerical_value)