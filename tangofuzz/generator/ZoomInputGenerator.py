from . import debug, warning, critical

from generator import InputGeneratorBase
from networkio   import ChannelFactoryBase
from statemanager import StateBase
from input import InputBase, ZoomInput
from interaction import (KillInteraction, DelayInteraction, ActivateInteraction,
                        ResetKeysInteraction, ReachInteraction, MoveInteraction,
                        RespawnInteraction)
from random import Random
from mutator import ZoomMutator
from enum import Enum, auto
import asyncio
from collections import OrderedDict
from typing import Sequence, Tuple
import networkx as nx
from common import Suspendable

class InterruptReason(Enum):
    NO_REASON = auto()
    ATTACKER_VALID = auto()
    LAVA_PIT = auto()
    PLAYER_DEATH = auto()
    SPECIAL_OBJECT = auto()

    NUM_REASONS = auto()

class ZoomInputGenerator(InputGeneratorBase):
    PREEMPTION_LISTS = OrderedDict([
        (0, (InterruptReason.PLAYER_DEATH,)),
        (10, (InterruptReason.ATTACKER_VALID,)),
        (20, (InterruptReason.SPECIAL_OBJECT,)),
    ])
    PREEMPTION_REPEAT_THRESHOLD = 100

    def __init__(self, startup: str, seed_dir: str, protocol: str):
        super().__init__(startup, seed_dir, protocol)
        self._reason = InterruptReason.NO_REASON
        self._feedback_task = None
        self._attacked_location = None
        self._resumable_keys = []

    def generate(self, state: StateBase, entropy: Random) -> InputBase:
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

    def generate_follow_path(self, \
            path: Sequence[Tuple[StateBase, StateBase, InputBase]]):
        for src, dst, inp in path:
            dst_x, dst_y, dst_z = map(lambda c: getattr(dst._struct, c), ('x', 'y', 'z'))
            condition = lambda: dst._sman._tracker.current_state == dst
            yield ReachInteraction(src, (dst_x, dst_y, dst_z), condition=condition)

    def generate_kill_sequence(self, state, from_location=None):
        yield KillInteraction(state)
        if from_location:
            destination_state = min(
                    filter(lambda x: nx.has_path(
                            state._sman.state_machine._graph,
                            state._sman._tracker._entry_state,
                            x),
                        state._sman.state_machine._graph.nodes), \
                    key=lambda s: ReachInteraction.l2_distance(
                            (from_location[0], from_location[1]),
                            (s._struct.x, s._struct.y)))
            path = next(state._sman._loader.live_path_gen(destination_state, state._sman))
            yield from self.generate_follow_path(path)

    @classmethod
    def get_reason_priority(cls, reason):
        for p, rs in cls.PREEMPTION_LISTS.items():
            if reason in rs:
                return p
        else:
            return p + 1

    async def feedback(self):
        self._target_task = asyncio.get_running_loop().main_task
        self._target_task.coro.set_callbacks(
            suspend_cb=self.suspend_cb, resume_cb=self.resume_cb
        )

        pending = 0
        while True:
            struct = self._state._sman._tracker._reader.struct
            reasons = []
            if struct.playerstate > 0:
                reasons.append(InterruptReason.PLAYER_DEATH)
            if struct.attacker_valid:
                if not self._attacked_location:
                    self._attacked_location = (struct.x, struct.y)
                if ReachInteraction.l2_distance(
                        (struct.x, struct.y),
                        (struct.attacker_x, struct.attacker_y)) < 1000**2:
                    reasons.append(InterruptReason.ATTACKER_VALID)
            if struct.canactivate:
                reasons.append(InterruptReason.SPECIAL_OBJECT)

            if reasons:
                pending += 1
            else:
                pending = 0
            if pending > self.PREEMPTION_REPEAT_THRESHOLD:
                # force resending the interrupt
                warning("Resending interrupt!")
                self._reason = InterruptReason.NO_REASON
                pending = 0

            preemptive_reason = None
            for r in reasons:
                if self.get_reason_priority(r) < self.get_reason_priority(self._reason) and \
                        self.get_reason_priority(r) < self.get_reason_priority(preemptive_reason):
                    preemptive_reason = r

            if preemptive_reason:
                self._reason = preemptive_reason
                warning(f"Interrupting main task due to {self._reason=}")
                asyncio.create_task(self.handle_reason(preemptive_reason))
            elif (self._reason == InterruptReason.ATTACKER_VALID and \
                    not struct.attacker_valid) or \
                 (self._reason == InterruptReason.PLAYER_DEATH and \
                    struct.playerstate == 0) or \
                 (self._reason == InterruptReason.SPECIAL_OBJECT and \
                    not struct.canactivate):
                warning("Clearing reason, looking for new reasons")
                self._reason = InterruptReason.NO_REASON
                self._attacked_location = None

            await DelayInteraction(0.1).perform(self._state._sman._loader._channel)

    async def handle_reason(self, reason):
        async def internal_handle_reason():
            main_task = self._target_task
            self._target_task = asyncio.current_task()

            try:
                if self.get_reason_priority(reason) > self.get_reason_priority(self._reason):
                    warning(f"My {reason=} is less important than existing {self._reason=}. Ignoring")
                    return True

                channel = self._state._sman._loader._channel
                if reason == InterruptReason.PLAYER_DEATH:
                    # FIXME is this the correct way to handle this?
                    try:
                        main_task.coro.suspend()
                        await RespawnInteraction(self._state).perform(channel)
                        await ResetKeysInteraction().perform(channel)
                    finally:
                        main_task.coro.resume()
                    main_task.cancel()
                    return False
                elif reason == InterruptReason.ATTACKER_VALID:
                    try:
                        main_task.coro.suspend()
                        await KillInteraction(self._state).perform(channel)
                    finally:
                        main_task.coro.resume()
                    return True
                elif reason == InterruptReason.SPECIAL_OBJECT:
                    try:
                        main_task.coro.suspend()
                        await ActivateInteraction().perform(channel)
                    finally:
                        main_task.coro.resume()
                    return True
            except asyncio.CancelledError:
                warning(f"Interrupt handler got cancelled! Propagating to {main_task=}")
                main_task.cancel()
            finally:
                warning(f"Finished processing {reason=}")
                self._target_task = main_task

        inter_task = asyncio.current_task()
        inter_task.suspendable_ancestors = []
        inter_task.coro = Suspendable(internal_handle_reason())
        inter_task.coro.set_callbacks(
            suspend_cb=self.suspend_cb, resume_cb=self.resume_cb
        )
        await inter_task.coro

    def suspend_cb(self):
        res = self._state._sman._loader._channel.sync_clear()
        self._resumable_keys.extend(res)

    def resume_cb(self):
        self._state._sman._loader._channel.sync_send(self._resumable_keys)
        self._resumable_keys.clear()
