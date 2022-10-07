from profiler import ProfileValue
from collections import deque
from statistics import mean

class ProfileValueMean(ProfileValue):
    def __init__(self, name, samples=10, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._maxlen = samples
            if self._maxlen > 0:
                self._samples = deque(maxlen=self._maxlen)
                self.___call_internal___ = self.___call_lastn___
                self._calculate = lambda: mean(self._samples) if self._samples else None
            else:
                self._count = 0
                self._sum = 0
                self.___call_internal___ = self.___call_total___
                self._calculate = lambda: self._sum / self._count if self._count != 0 else None

    def ___call___(self, obj):
        return self.___call_internal___(obj)

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