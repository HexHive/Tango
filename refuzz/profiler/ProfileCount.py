from profiler import ProfileValue

class ProfileCount(ProfileValue):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        if not self._init_called:
            self._count = 0

    def __call__(self, obj):
        self._count += obj
        return obj

    @staticmethod
    def _format(num):
        for unit in ['','K']:
            if abs(num) < 1000.0:
                return "%.1f%s" % (num, unit)
            num /= 1000.0
        return "%.1f%s" % (num, 'M')

    @property
    def value(self):
        return self._format(self._count)
