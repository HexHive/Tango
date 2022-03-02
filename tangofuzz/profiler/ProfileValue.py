from profiler import ProfilerBase

class ProfileValue(ProfilerBase):
    def __call__(self, obj):
        self._value = obj
        return obj

    @property
    def value(self):
        return self._value