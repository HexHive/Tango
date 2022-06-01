from . import debug, warning

from generator import InputGeneratorBase
from networkio   import ChannelFactoryBase
from statemanager import StateBase
from input import InputBase, ZoomInput
from interaction import (KillInteraction, DelayInteraction, ActivateInteraction,
                        ResetKeysInteraction)
from random import Random
from mutator import ZoomMutator
from enum import Enum, auto
import asyncio
from collections import OrderedDict

class InterruptReason(Enum):
    NO_REASON = auto()
    ATTACKER_VALID = auto()
    LAVA_PIT = auto()
    PLAYER_DEATH = auto()

    NUM_REASONS = auto()

class ZoomInputGenerator(InputGeneratorBase):
    PREEMPTION_LISTS = OrderedDict([
        (0, (InterruptReason.PLAYER_DEATH,)),
        (10, (InterruptReason.ATTACKER_VALID,)),
    ])

    def __init__(self, startup: str, seed_dir: str, protocol: str):
        super().__init__(startup, seed_dir, protocol)
        self._reason = InterruptReason.NO_REASON
        self._feedback_task = None

    def generate(self, state: StateBase, entropy: Random) -> InputBase:
        if self._reason == InterruptReason.NO_REASON:
            if self._feedback_task is None or self._feedback_task.done():
                if self._feedback_task is not None:
                    # prev_result = await self._feedback_task
                    pass
                self._state = state
                self._feedback_task = asyncio.create_task(self.feedback())

            candidate = state.get_escaper()
            if candidate is None:
                out_edges = list(state.out_edges)
                if out_edges:
                    _, dst, data = entropy.choice(out_edges)
                    candidate = data['minimized']
                else:
                    in_edges = list(state.in_edges)
                    if in_edges:
                        _, dst, data = entropy.choice(in_edges)
                        candidate = data['minimized']
                    elif self.seeds:
                        candidate = entropy.choice(self.seeds)
                    else:
                        candidate = ZoomInput()

            return ZoomMutator(entropy, state)(candidate)
        else:
            try:
                if self._reason == InterruptReason.ATTACKER_VALID:
                    return ZoomInput((KillInteraction(state),))
                elif self._reason == InterruptReason.PLAYER_DEATH:
                    return ZoomInput((ActivateInteraction(), ResetKeysInteraction()))
            finally:
                pass
                # self._reason = InterruptReason.NO_REASON

    @classmethod
    def get_reason_priority(cls, reason):
        for p, rs in cls.PREEMPTION_LISTS.items():
            if reason in rs:
                return p
        else:
            return p + 1

    async def feedback(self):
        while True:
            struct = self._state._sman._tracker._reader.struct
            reasons = []
            if struct.playerstate > 0:
                reasons.append(InterruptReason.PLAYER_DEATH)
            if struct.attacker_valid:
                reasons.append(InterruptReason.ATTACKER_VALID)

            preemptive_reason = None
            for r in reasons:
                if self.get_reason_priority(r) < self.get_reason_priority(self._reason):
                    preemptive_reason = r

            if preemptive_reason:
                self._reason = preemptive_reason
                task = getattr(asyncio.get_event_loop(), '_executing_task', None)
                if task is not None and not task.done():
                    warning(f"Sending interrupt now! {self._reason = }")
                    task.cancel()

            elif (self._reason == InterruptReason.ATTACKER_VALID and \
                    not struct.attacker_valid) or \
                 (self._reason == InterruptReason.PLAYER_DEATH and \
                    struct.playerstate == 0):
                warning("Clearing reason, looking for new reasons")
                self._reason = InterruptReason.NO_REASON

            await DelayInteraction(0.1).perform(self._state._sman._loader._channel)