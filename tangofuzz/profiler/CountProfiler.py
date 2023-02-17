from profiler import ValueProfiler

class CountProfiler(ValueProfiler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
