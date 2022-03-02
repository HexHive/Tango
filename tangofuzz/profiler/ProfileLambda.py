from profiler import ProfilerBase

class ProfileLambda(ProfilerBase):
    def __call__(self, obj):
        self._obj = obj
        return self._obj()

    @property
    def value(self):
        return self._obj()