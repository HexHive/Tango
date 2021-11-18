from profiler import ProfileValue, ProfiledObjects as objects
from collections import deque
from statistics import mean

class ProfileValueMean(ProfileValue):
    def __init__(self, name, samples=10, **kwargs):
        if name in objects:
            return
        super().__init__(name, **kwargs)
        self._maxlen = samples
        if self._maxlen > 0:
            self._samples = deque(maxlen=self._maxlen)
            self.___call___ = self.___call_lastn___
            self._calculate = lambda: mean(self._samples) if self._samples else None
        else:
            self._count = 0
            self._sum = 0
            self.___call___ = self.___call_total___
            self._calculate = lambda: self._sum / self._count if self._count != 0 else None

    def __call__(self, obj):
        return self.___call___(obj)

    def ___call_total___(self, obj):
        self._sum += obj
        self._count += 1
        return obj

    def ___call_lastn___(self, obj):
        self._samples.append(obj)
        return obj

    @property
    def value(self):
        return self.truncate(self._calculate())