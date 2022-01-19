from __future__ import annotations
from interaction import InteractionBase
from networkio   import ChannelBase
from typing      import Sequence
from time        import sleep
from profiler    import ProfileFrequency, ProfileEvent
from math        import isclose

class DelayInteraction(InteractionBase):
    def __init__(self, time: float):
        self._time = time
        # a delay longer than 5 seconds is probably useless, but we'll keep this
        # configurable
        self._maxdelay = 5

    @ProfileEvent("perform_interaction")
    @ProfileFrequency("interactions", period=1)
    def perform(self, channel: ChannelBase):
        sleep(self._time * channel._timescale)

    def __eq__(self, other: DelayInteraction):
        return isinstance(other, DelayInteraction) and \
            isclose(self._time, other._time, rel_tol=0.01)

    def mutate(self, mutator, entropy):
        a, b = self._time.as_integer_ratio()

        b = mutator.mutate_int(b, entropy) or 1
        a = (mutator.mutate_int(a, entropy) % b) * self._maxdelay

        self._time = a / b
