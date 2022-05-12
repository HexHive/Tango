from __future__ import annotations
from interaction import InteractionBase
from networkio   import X11Channel
import time

class ActivateInteraction(InteractionBase):
    def __init__(self):
        pass

    def perform(self, channel: X11Channel):
        key = "space"
        channel.send(key, release=False)
        time.sleep(0.1)
        channel.send(key, release=True)

    def mutate(self, mutator, entropy):
        pass

    def __eq__(self, other: ActivateInteraction) -> bool:
        return isinstance(other, ActivateInteraction)