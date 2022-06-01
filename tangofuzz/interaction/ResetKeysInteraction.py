from __future__ import annotations
from interaction import InteractionBase
from networkio   import X11Channel

class ResetKeysInteraction(InteractionBase):
    def __init__(self):
        pass

    async def perform(self, channel: X11Channel):
        await channel.clear()

    def mutate(self, mutator, entropy):
        pass

    def __eq__(self, other: ResetKeysInteraction) -> bool:
        return isinstance(other, ResetKeysInteraction)

    def __repr__(self):
        return 'reset keys'