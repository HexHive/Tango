class LoadedException(RuntimeError):
    def __init__(self, ex, payload=None):
        self._payload = payload

    @property
    def payload(self):
        return self._payload

class StabilityException(RuntimeError):
    pass

class ChannelSetupException(RuntimeError):
    pass

class ChannelBrokenException(RuntimeError):
    pass