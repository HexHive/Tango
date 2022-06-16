from .. import warning

from statemanager import StateBase, StrategyBase, StateMachine
from random import Random
from profiler import ProfileValue
from input import InputBase, ZoomInput
from interaction import (KillInteraction, DelayInteraction, ActivateInteraction,
                        ResetKeysInteraction, ReachInteraction, RespawnInteraction)
import networkx as nx
from common import Suspendable
from collections import OrderedDict
import asyncio
from enum import Enum, auto

class ZoomStrategy(StrategyBase):
    def __init__(self, entropy: Random, limit: int=50, **kwargs):
        super().__init__(**kwargs)
        self._entropy = entropy
        self._counter = 0
        self._limit = limit
        self._invalid_states = set()
        self._target_state = self._entry
        self._cycles = 0
        self._preemptor = ZoomPreemptor(self._entry._sman._tracker._reader,
                                        self._entry._sman._loader,
                                        self._entry)

    @staticmethod
    def _calculate_edge_weight(src, dst, data):
        return ReachInteraction.l2_distance(
            (src._struct.player_location.x, src._struct.player_location.y),
            (dst._struct.player_location.x, dst._struct.player_location.y)
        )

    def _recalculate_target(self):
        if all(x._cycle == self._cycles for x in self._sm._graph.nodes):
            self._cycles += 1
        self._counter = 0
        ProfileValue('strat_counter')(self._counter)
        filtered = [x for x in self._sm._graph.nodes if x not in self._invalid_states]
        distances = nx.shortest_path_length(self._sm._graph, self._entry, \
                                        weight=self._calculate_edge_weight, \
                                        method='bellman-ford')
        self._target_state = next(x for x in sorted(
            distances.keys(),
            key=lambda x: distances[x],
            reverse=True
        ) if x._cycle < self._cycles)
        self._target_state._cycle = self._cycles

    def step(self) -> bool:
        should_reset = False
        if self._counter == 0:
            should_reset = True
            self._recalculate_target()
        else:
            ProfileValue('strat_counter')(self._counter)
        self._counter += 1
        self._counter %= self._limit
        return should_reset

    @property
    def target_state(self) -> StateBase:
        return self._target_state

    @property
    def target(self) -> StateBase:
        return self._target_state

    def update_state(self, state: StateBase, invalidate: bool = False, is_new: bool = False):
        if invalidate:
            self._invalid_states.add(state)
            if self._target_state == state:
                self._recalculate_target()
        else:
            self._invalid_states.discard(state)
            if is_new or not hasattr(state, '_cycle'):
                state._cycle = self._cycles - 1

    def update_transition(self, source: StateBase, destination: StateBase, input: InputBase, invalidate: bool = False):
        src_x, src_y, src_z = map(lambda c: getattr(source._struct.player_location, c), ('x', 'y', 'z'))
        dst_x, dst_y, dst_z = map(lambda c: getattr(destination._struct.player_location, c), ('x', 'y', 'z'))
        if dst_z > src_z or abs(src_z - dst_z) <= 16:
            # FIXME this interaction is not used, since it is replaced by the loader
            inp = ZoomInput([ReachInteraction(destination, (src_x, src_y, src_z))])
            self._sm.update_transition(destination, source, inp)
            warning(f"Added reverse transition from {destination} to {source}")

class InterruptReason(Enum):
    NO_REASON = auto()
    ATTACKER_VALID = auto()
    LAVA_PIT = auto()
    PLAYER_DEATH = auto()
    SPECIAL_OBJECT = auto()

    NUM_REASONS = auto()

class ZoomPreemptor:
    PREEMPTION_LISTS = OrderedDict([
        (0, (InterruptReason.PLAYER_DEATH,)),
        (10, (InterruptReason.ATTACKER_VALID,)),
        (20, (InterruptReason.SPECIAL_OBJECT,)),
    ])
    PREEMPTION_REPEAT_THRESHOLD = 100
    ATTACKER_MAX_TRIGGER_DISTANCE = 1000

    def __init__(self, reader, loader, entry_state):
        self._reader = reader
        self._loader = loader
        # FIXME state should not be passed around like this
        self._state = entry_state
        self._reason = InterruptReason.NO_REASON
        self._attacked_location = None
        self._resumable_keys = []
        self._feedback_task = asyncio.create_task(self.feedback())

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
            struct = self._reader.struct
            reasons = []
            if struct.playerstate > 0:
                reasons.append(InterruptReason.PLAYER_DEATH)
            if struct.attacker_valid:
                if not self._attacked_location:
                    self._attacked_location = (struct.player_location.x, struct.player_location.y)
                if ReachInteraction.l2_distance(
                        (struct.player_location.x, struct.player_location.y),
                        (struct.attacker_location.x, struct.attacker_location.y)) < \
                        self.ATTACKER_MAX_TRIGGER_DISTANCE**2:
                    reasons.append(InterruptReason.ATTACKER_VALID)
            if struct.can_activate:
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
                    not struct.can_activate):
                warning("Clearing reason, looking for new reasons")
                self._reason = InterruptReason.NO_REASON
                self._attacked_location = None

            await DelayInteraction(0.1).perform(self._loader._channel)

    async def handle_reason(self, reason):
        async def internal_handle_reason():
            main_task = self._target_task
            self._target_task = asyncio.current_task()

            try:
                if self.get_reason_priority(reason) > self.get_reason_priority(self._reason):
                    warning(f"My {reason=} is less important than existing {self._reason=}. Ignoring")
                    return True

                channel = self._loader._channel
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
        res = self._loader.channel.sync_clear()
        self._resumable_keys.extend(res)

    def resume_cb(self):
        self._loader.channel.sync_send(self._resumable_keys)
        self._resumable_keys.clear()
