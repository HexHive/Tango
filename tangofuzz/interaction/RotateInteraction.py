from __future__ import annotations
from . import debug

from interaction import InteractionBase, MoveInteraction, DelayInteraction
from networkio   import X11Channel
import time
from math import isclose
from copy import deepcopy

import asyncio

class RotateInteraction(InteractionBase):
    def __init__(self, current_state, target_angle: float, tolerance: float=2.0):
        self._target = float(target_angle)
        self._struct = current_state.state_manager._tracker._reader.struct
        self._tol = tolerance

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ('_struct'):
                setattr(result, k, deepcopy(v, memo))
            else:
                setattr(result, k, getattr(self, k))
        return result

    @staticmethod
    def short_angle(angle):
        return sorted((angle, angle - 360, angle + 360), key=lambda x: abs(x))[0]

    async def perform_internal(self, channel: X11Channel, move: str=None):
        last = None
        while not isclose(delta := self.short_angle(self._struct.player_angle - self._target), 0, abs_tol=self._tol):
            debug(f"Still far from target {delta=}")

            if move and not last:
                debug("Stopping movement while rotating")
                await MoveInteraction(move, stop=True).perform(channel)
            d = self._target - self._struct.player_angle
            if self.short_angle(d) > 0:
                current = MoveInteraction('rotate_left')
            else:
                current = MoveInteraction('rotate_right')
            if last and last != current:
                debug("Switching rotation direction")
                await last.stop(channel)
            await current.perform(channel)
            last = current
            await DelayInteraction(0.03).perform(channel)

        if last:
            await last.stop(channel)
            if move:
                debug("Resuming movement")
                await MoveInteraction(move).perform(channel)

    def mutate(self, mutator, entropy):
        a, b = self._target.as_integer_ratio()

        b = mutator.mutate_int(b, entropy) or 1
        a = (mutator.mutate_int(a, entropy) % b) * 360

        self._target = a / b

    def __eq__(self, other: RotateInteraction) -> bool:
        return isinstance(other, RotateInteraction) and \
            isclose(self.short_angle(self._target - other._target), 0, abs_tol=self._tol)

    def __repr__(self):
        return 'rotate'