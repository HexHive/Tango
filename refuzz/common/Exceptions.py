class LoadedException(RuntimeError):
    def __new__(cls, ex, payload=None):
        if isinstance(ex, LoadedException):
            new = super(LoadedException, cls).__new__(cls)
            new._ex = ex._ex
            new._payload = payload
            return new
        else:
            new = super(LoadedException, cls).__new__(cls)
            new._ex = ex
            new._payload = payload
            return new

    @property
    def payload(self):
        return self._payload

    @property
    def exception(self):
        return self._ex

class StabilityException(RuntimeError):
    pass

class StatePrecisionException(RuntimeError):
    pass

class ChannelSetupException(RuntimeError):
    pass

class ChannelBrokenException(RuntimeError):
    pass

class ChannelTimeoutException(RuntimeError):
    pass

class ProcessTerminatedException(RuntimeError):
    def __init__(self, msg, exitcode, *args):
        super().__init__(msg, *args)
        self._exitcode = exitcode

class ProcessCrashedException(ProcessTerminatedException):
    pass

class StateNotReproducibleException(RuntimeError):
    def __init__(self, msg, faulty_state, *args):
        super().__init__(msg, *args)
        self._faulty_state = faulty_state