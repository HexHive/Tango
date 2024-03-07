class NotSyncedException(RuntimeError):
    pass

class LoadedException(RuntimeError):
    def __new__(cls, ex, payload=None):
        if isinstance(ex, LoadedException):
            new = super(LoadedException, cls).__new__(cls)
            new._ex = ex._ex
            new._payload_ctor = payload
            return new
        else:
            new = super(LoadedException, cls).__new__(cls)
            new._ex = ex
            new._payload_ctor = payload
            return new

    @property
    def payload(self):
        return self._payload_ctor()

    @payload.setter
    def payload(self, value):
        self._payload_ctor = lambda: value

    @property
    def exception(self):
        return self._ex

class StabilityException(RuntimeError):
    def __init__(self, msg, current_state, expected_state):
        super().__init__(msg)
        self.current_state = current_state
        self.expected_state = expected_state

class StatePrecisionException(RuntimeError):
    pass

class ChannelSetupException(RuntimeError):
    pass

class ChannelBrokenException(RuntimeError):
    CHANNEL_SHUTDOWN = 0
    CHANNEL_RESET = 1
    CHANNEL_CLOSE = 2
    def __init__(self, *args: object, exitcode=None) -> None:
        super().__init__(*args)
        self._exitcode = exitcode

class ChannelTimeoutException(RuntimeError):
    pass

class ProcessTerminatedException(RuntimeError):
    def __init__(self, msg, exitcode=None, signum=None):
        super().__init__(msg)
        self._exitcode = exitcode
        self._signum = signum

class ProcessCrashedException(ProcessTerminatedException):
    pass

class StateNotReproducibleException(RuntimeError):
    def __init__(self, msg, faulty_state):
        super().__init__(msg)
        self.faulty_state = faulty_state

class PathNotReproducibleException(RuntimeError):
    def __init__(self, msg, faulty_path):
        super().__init__(msg)
        self.faulty_path = faulty_path