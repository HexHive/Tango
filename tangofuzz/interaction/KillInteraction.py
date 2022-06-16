from __future__ import annotations
from . import debug, warning

from interaction import (InteractionBase, RotateInteraction, ReachInteraction,
                        ShootInteraction, MoveInteraction)
from networkio   import X11Channel
from math import atan2
from copy import deepcopy

class KillInteraction(InteractionBase):
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

    async def perform_internal(self, channel: X11Channel):
        if self._struct.attacker_valid:
            restore = await channel.clear()
        else:
            restore = set()

        limit = 10
        while self._struct.attacker_valid and limit > 0:
            angle = (atan2(
                    self._struct.attacker_y - self._struct.y,
                    self._struct.attacker_x - self._struct.x
                ) * 180 / 3.14159265) % 360

            debug(f"Adjusting rotation to target {self._struct.angle=} {angle=}")
            if self._struct.ammo[1] > 0:
                weapon = 3
                await RotateInteraction(self._state, angle).perform(channel)
            elif self._struct.ammo[0] > 0:
                weapon = 2
                await RotateInteraction(self._state, angle).perform(channel)
            else:
                weapon = 1
                await ReachInteraction(self._state,
                    (self._struct.attacker_x, self._struct.attacker_y)).perform(channel)

            await ShootInteraction(weapon).perform(channel)
            limit -= 1
            if not self._struct.attacker_valid:
                break

        # FIXME does this result in changing the shm struct?
        self._struct.attacker_valid = False

        # restore previous movement, if any
        await channel.send(restore, down=True)

    def mutate(self, mutator, entropy):
        pass

    def __eq__(self, other: KillInteraction) -> bool:
        return isinstance(other, KillInteraction)

    def __repr__(self):
        return 'kill'