from __future__ import annotations
from interaction import InteractionBase
from networkio   import X11Channel
import time

class ShootInteraction(InteractionBase):
    def __init__(self, weapon: int=2):
        self._weapon = weapon

    def perform(self, channel: X11Channel):
        seq = (str(self._weapon), "Control_L")
        for key in seq:
            channel.send(key, release=False)
            time.sleep(0.1)
            channel.send(key, release=True)

    def mutate(self, mutator, entropy):
        pass

    def __eq__(self, other: ShootInteraction) -> bool:
        return isinstance(other, ShootInteraction)