from __future__ import annotations
from interaction import InteractionBase, DelayInteraction, ActivateInteraction
from networkio   import X11Channel
from copy import deepcopy

class RespawnInteraction(InteractionBase):
    def __init__(self, current_state):
        self._state = current_state
        self._struct = current_state.state_manager._tracker._reader.struct

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ('_state', '_struct'):
                setattr(result, k, deepcopy(v, memo))
            else:
                setattr(result, k, getattr(self, k))
        return result

    async def perform(self, channel: X11Channel):
        await ActivateInteraction().perform(channel)
        tracker = self._state._sman.state_tracker
        while tracker.current_state != tracker.entry_state:
            await DelayInteraction(0.01).perform(channel)
        await self._state._sman.reset_state()

    def mutate(self, mutator, entropy):
        pass

    def __eq__(self, other: RespawnInteraction) -> bool:
        return isinstance(other, RespawnInteraction)

    def __repr__(self):
        return 'respawn'