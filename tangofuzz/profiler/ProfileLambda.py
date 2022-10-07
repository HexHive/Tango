from profiler import ProfilerBase

class ProfileLambda(ProfilerBase):
    def ___call___(self, obj):
        self._obj = obj
        return self._obj()

    @property
    def value(self):
        return self._obj()