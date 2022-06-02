from __future__ import annotations
from . import debug

from common import ChannelTimeoutException
from interaction import InteractionBase, ReachInteraction
from networkio   import X11Channel
from math import isclose, atan2
from copy import deepcopy
from collections import deque
from statistics import variance
from enum import Enum, auto

class GoToInteraction(InteractionBase):
    def __init__(self, current_state, target_location: tuple, stop_at_target: bool=True, tolerance: float=5.0):
        self._target = target_location
        self._struct = current_state.state_manager._tracker._reader.struct
        self._state = current_state
        self._stop = stop_at_target
        self._tol = tolerance
        self._target_state = min(self._state._sman.state_machine._graph.nodes, \
                    key=lambda s: ReachInteraction.l2_distance(
                            (target_location[0], target_location[1]),
                            (s._struct.x, s._struct.y)))

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
        await self._state._sman._loader.load_state(self._target_state, self._state._sman, update=False)
        await ReachInteraction(self._state, self._target, self._stop, self._tol).perform(channel)

    def mutate(self, mutator, entropy):
        #FIXME mutate maybe
        pass

    def __eq__(self, other: GoToInteraction) -> bool:
        return isinstance(other, GoToInteraction) and \
            isclose(ReachInteraction.l2_distance(
                    (other._target[0], other._target[1]),
                    (self._target[0], self._target[1])
                ), 0, abs_tol=self._tol**2)

    def __repr__(self):
        return f'go to {self._target}'