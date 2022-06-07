from __future__ import annotations
from . import debug

from common import ChannelTimeoutException
from interaction import (InteractionBase, DelayInteraction,
                        MoveInteraction, RotateInteraction,
                        ActivateInteraction)
from networkio   import X11Channel
from math import isclose, atan2
from copy import deepcopy
from collections import deque
from statistics import variance
from enum import Enum, auto

class RecoveringAttempt(Enum):
    NOT_RECOVERING = auto()
    ACTIVATE = auto()
    JITTER_LEFT = auto()
    JITTER_RIGHT = auto()
    ALL_ATTEMPTS = auto()

class ReachInteraction(InteractionBase):
    MAX_STRAFE_DISTANCE = 50
    DISTANCE_SAMPLE_COUNT = 20
    DISTANCE_SAMPLE_VARIANCE = 400
    MAX_RETRIES = 1

    def __init__(self, current_state, target_location: tuple, stop_at_target: bool=True, tolerance: float=5.0, condition=None):
        self._target = target_location
        self._struct = current_state.state_manager._tracker._reader.struct
        self._state = current_state
        self._stop = stop_at_target
        self._tol = tolerance
        self._con = condition

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

    @staticmethod
    def l2_distance(a, b):
        return sum(map(lambda x: (x[0] - x[1])**2, zip(a, b)))

    async def perform(self, channel: X11Channel):
        last = None
        strafing = False
        d_samples = deque(maxlen=self.DISTANCE_SAMPLE_COUNT)
        recovering = RecoveringAttempt.NOT_RECOVERING
        retries = 0
        try:
            while not (isclose(distancesq := self.l2_distance(
                                (self._struct.x, self._struct.y),
                                (self._target[0], self._target[1])
                            ), 0, abs_tol=self._tol**2) and \
                        (self._con() if self._con else True)):
                angle = (atan2(
                                self._target[1] - self._struct.y,
                                self._target[0] - self._struct.x
                            ) * 180 / 3.14159265) % 360
                debug(f"Still far from target {distancesq=} {angle=}")
                d_samples.append(distancesq)
                if len(d_samples) == self.DISTANCE_SAMPLE_COUNT:
                    if variance(d_samples) < self.DISTANCE_SAMPLE_VARIANCE:
                        if recovering == RecoveringAttempt.ALL_ATTEMPTS or \
                                retries >= self.MAX_RETRIES:
                            raise ChannelTimeoutException("Unable to reach target location")
                        else:
                            recovering = RecoveringAttempt(recovering.value + 1)
                    elif recovering != RecoveringAttempt.NOT_RECOVERING:
                        retries += 1
                        recovering = RecoveringAttempt.NOT_RECOVERING

                if recovering == RecoveringAttempt.ACTIVATE:
                    await ActivateInteraction().perform(channel)
                    # wait for door to open or something
                    await DelayInteraction(0.5).perform(channel)
                elif recovering == RecoveringAttempt.JITTER_LEFT:
                    await MoveInteraction('strafe_left', duration=1).perform(channel)
                elif recovering == RecoveringAttempt.JITTER_RIGHT:
                    await MoveInteraction('strafe_right', duration=1).perform(channel)

                if distancesq < self.MAX_STRAFE_DISTANCE ** 2:
                    strafing = True
                    delta = RotateInteraction.short_angle(angle - self._struct.angle) % 360
                    if delta < 20 or 340 < delta:
                        current = MoveInteraction('forward')
                    elif 20 <= delta <= 160:
                        current = MoveInteraction('strafe_left')
                    elif 160 < delta < 200:
                        current = MoveInteraction('backward')
                    elif 200 <= delta <= 340:
                        current = MoveInteraction('strafe_right')
                else:
                    debug(f"Adjusting rotation to target {self._struct.angle=} {angle=}")
                    await RotateInteraction(self._state, angle).perform(channel, move=last._dir if last else None)
                    if strafing and last._dir != 'forward':
                        debug(f'Releasing {last=}')
                        await last.stop(channel)
                        last = None
                        strafing = False
                    current = MoveInteraction('forward')
                if current != last:
                    if last:
                        debug(f'Releasing {last=}')
                        await last.stop(channel)
                    debug(f'Performing {current=}')
                    await current.perform(channel)
                    last = current
                await DelayInteraction(0.03).perform(channel)
        finally:
            if last and (self._stop or strafing):
                await last.stop(channel)

    def mutate(self, mutator, entropy):
        #FIXME mutate maybe
        pass

    def __eq__(self, other: ReachInteraction) -> bool:
        return isinstance(other, ReachInteraction) and \
            isclose(self.l2_distance(
                    (other._target[0], other._target[1]),
                    (self._target[0], self._target[1])
                ), 0, abs_tol=self._tol**2)

    def __repr__(self):
        return f'bee-line to {self._target}'