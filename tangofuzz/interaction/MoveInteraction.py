from __future__ import annotations
from interaction import InteractionBase
from networkio   import X11Channel
from collections import OrderedDict


class MoveInteraction(InteractionBase):
    DIRECTION_KEY_MAP = OrderedDict([
        ("forward", {"Up",}),
        ("rotate_left", {"Left",}),
        ("rotate_right", {"Right",}),
        ("strafe_left", {"Alt_L", "Left"}),
        ("strafe_right", {"Alt_L", "Right"}),
        ("backward", {"Down",}),
    ])
    DIRECTION_CLOBBERS = (
        ("forward", "backward"),
        ("strafe_left", "strafe_right", "rotate_left", "rotate_right")
    )

    def __init__(self, direction: str, stop: bool=False):
        self._dir = direction
        self._stop = stop

    def clobbered_keys(self, dir):
        clobbered = set()
        for clobbers in self.DIRECTION_CLOBBERS:
            if dir in clobbers:
                for c in clobbers:
                    if c != dir:
                        clobbered |= self.DIRECTION_KEY_MAP[c]
        return clobbered

    async def perform(self, channel: X11Channel):
        keys = self.DIRECTION_KEY_MAP.get(self._dir, ())
        await channel.send(keys,
            down=not self._stop,
            clobbers=self.clobbered_keys(self._dir)
        )

    async def stop(self, channel: X11Channel):
        keys = self.DIRECTION_KEY_MAP.get(self._dir, ())
        await channel.send(keys,
            down=False,
            clobbers=self.clobbered_keys(self._dir)
        )

    def mutate(self, mutator, entropy):
        self._dir = entropy.choices(
                list(self.DIRECTION_KEY_MAP.keys()),
                cum_weights=(5, 8, 11, 12, 13, 14)
            )[0]
        self._stop = entropy.choice((True, False))

    def __eq__(self, other: MoveInteraction) -> bool:
        return isinstance(other, MoveInteraction) and \
            other._dir == self._dir and other._stop == self._stop

    def __repr__(self):
        return f'{"move" if not self._stop else "stop"} {self._dir}'