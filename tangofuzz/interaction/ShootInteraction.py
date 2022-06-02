from __future__ import annotations
from interaction import InteractionBase, DelayInteraction
from networkio   import X11Channel
import time

class ShootInteraction(InteractionBase):
    def __init__(self, weapon: int=2):
        self._weapon = weapon

    async def perform(self, channel: X11Channel):
        seq = (str(self._weapon), "Control_L")
        for key in seq:
            try:
                await channel.send(key, down=True)
                await DelayInteraction(0.1).perform(channel)
            finally:
                await channel.send(key, down=False)

    def mutate(self, mutator, entropy):
        pass

    def __eq__(self, other: ShootInteraction) -> bool:
        return isinstance(other, ShootInteraction)

    def __repr__(self):
        return 'shoot'