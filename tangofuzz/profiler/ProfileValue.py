from profiler import ProfilerBase

class ProfileValue(ProfilerBase):
    def ___call___(self, obj):
        self._value = obj
        return obj

    @property
    def value(self):
        return self._value