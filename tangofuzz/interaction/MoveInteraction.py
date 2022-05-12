from __future__ import annotations
from interaction import InteractionBase
from networkio   import X11Channel
from collections import OrderedDict


class MoveInteraction(InteractionBase):
    DIRECTION_KEY_MAP = OrderedDict([
        ("forward", ("Up",)),
        ("backward", ("Down",)),
        ("strafe_left", ("Alt_L", "Left")),
        ("strafe_right", ("Alt_L", "Right")),
        ("rotate_left", ("Left",)),
        ("rotate_right", ("Right",))
    ])

    def __init__(self, direction: str, stop: bool=False):
        self._dir = direction
        self._keys = self.DIRECTION_KEY_MAP.get(self._dir, ())
        self._release = stop

    def perform(self, channel: X11Channel):
        for key in self._keys:
            channel.send(key, release=self._release)

    def mutate(self, mutator, entropy):
        self._dir = entropy.choice(list(self.DIRECTION_KEY_MAP.keys()))
        self._keys = self.DIRECTION_KEY_MAP.get(self._dir, ())
        self._release = entropy.choice((True, False))

    def __eq__(self, other: MoveInteraction) -> bool:
        return isinstance(other, MoveInteraction) and other._dir == self._dir