from profiler import ProfilerBase, ProfiledObjects as objects

class ProfileValue(ProfilerBase):
    def __new__(cls, name, **kwargs):
        if isinstance(other := objects.get(name), ProfileValue):
            return other
        else:
            return super(ProfileValue, cls).__new__(cls)

    def __init__(self, name, **kwargs):
        if name in objects:
            return
        super().__init__(name, **kwargs)

    def __call__(self, obj):
        self._value = obj
        return obj

    @property
    def value(self):
        return self._value