from . import debug, warning

from generator import InputGeneratorBase
from networkio   import ChannelFactoryBase
from statemanager import StateBase
from input import InputBase, ZoomInput
from interaction import KillInteraction, DelayInteraction, ActivateInteraction
from random import Random
from mutator import ZoomMutator
from enum import Enum, auto
import asyncio

class ZoomInputGenerator(InputGeneratorBase):
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
                    return ZoomInput((ActivateInteraction(),))
            finally:
                self._reason = InterruptReason.NO_REASON

    async def feedback(self):
        while True:
            if self._reason == InterruptReason.NO_REASON:
                struct = self._state._sman._tracker._reader.struct
                if struct.playerstate > 0:
                    self._reason = InterruptReason.PLAYER_DEATH
                elif struct.attacker_valid:
                    self._reason = InterruptReason.ATTACKER_VALID

                if self._reason != InterruptReason.NO_REASON:
                    task = getattr(asyncio.get_event_loop(), '_executing_task', None)
                    if task is not None and not task.done():
                        task.cancel()
                        await DelayInteraction(10).perform(self._state._sman._loader._channel)

            await DelayInteraction(0.1).perform(self._state._sman._loader._channel)


class InterruptReason(Enum):
    NO_REASON = auto()
    ATTACKER_VALID = auto()
    LAVA_PIT = auto()
    PLAYER_DEATH = auto()

    NUM_REASONS = auto()
