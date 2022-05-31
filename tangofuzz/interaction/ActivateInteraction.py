from __future__ import annotations
from interaction import InteractionBase, DelayInteraction
from networkio   import X11Channel

class ActivateInteraction(InteractionBase):
    def __init__(self):
        pass

    async def perform(self, channel: X11Channel):
        key = "space"
        await channel.send(key, down=True)
        await DelayInteraction(0.1).perform(channel)
        await channel.send(key, down=False)

    def mutate(self, mutator, entropy):
        pass

    def __eq__(self, other: ActivateInteraction) -> bool:
        return isinstance(other, ActivateInteraction)

    def __repr__(self):
        return 'activate'