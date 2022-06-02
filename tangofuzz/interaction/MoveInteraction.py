from __future__ import annotations
from interaction import InteractionBase, DelayInteraction
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

    def __init__(self, direction: str, stop: bool=False, duration: float=None):
        self._dir = direction
        self._stop = stop
        self._duration = float(duration) if duration else None
        self._maxduration = 1.0

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
        if not self._stop and self._duration:
            await DelayInteraction(self._duration).perform(channel)
            await self.stop(channel)

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
        if not self._stop and self._duration:
            a, b = self._duration.as_integer_ratio()
            b = mutator.mutate_int(b, entropy) or 1
            a = (mutator.mutate_int(a, entropy) % b) * self._maxduration
            self._duration = a / b

    def __eq__(self, other: MoveInteraction) -> bool:
        return isinstance(other, MoveInteraction) and \
            other._dir == self._dir and other._stop == self._stop and \
            (self._duration and other._duration) # FIXME

    def __repr__(self):
        return (f'{"move" if not self._stop else "stop"} {self._dir}'
                f'{" for {} seconds".format(self._duration) if self._duration and not self._stop else ""}')