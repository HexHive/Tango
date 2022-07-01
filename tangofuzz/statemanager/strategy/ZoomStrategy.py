from .. import warning

from statemanager import StateBase, StrategyBase, StateMachine, ZoomState
from random import Random
from profiler import ProfileValue, ProfileLambda
from input import InputBase, ZoomInput
from interaction import (KillInteraction, DelayInteraction, ActivateInteraction,
                        ResetKeysInteraction, ReachInteraction, RespawnInteraction,
                        GoToInteraction)
import networkx as nx
from common import Suspendable
from collections import OrderedDict
import asyncio
from enum import Enum, auto
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

class ZoomStrategy(StrategyBase):
    INTERESTING_PICKUPS = (
        55, 56, # armors, green and blue
        62, 64, 63, 65, 67, 66, # keys
        87, 88, 89, 90, 91, 92, 93, # weapons
    )

    def __init__(self, entropy: Random, limit: int=50, **kwargs):
        super().__init__(**kwargs)
        self._entropy = entropy
        self._counter = 0
        self._limit = limit
        self._invalid_states = set()
        self._target_state = self._entry
        self._cycles = 0
        ProfileValue('strat_cycles')(self._cycles)

        self._seek = [] # it's a list because we may need them in order
        self._avoid = set()

        self._preemptor = ZoomPreemptor(self._entry._sman._tracker._reader,
                                        self._entry._sman._loader,
                                        self._entry,
                                        self)
        self._hull = None
        self._states = dict()
        self.update_state(self._entry)

        ProfileLambda('strat_seek')(lambda: str([x for y in self._seek for x in y._tags.items()]))
        ProfileLambda('strat_avoid')(lambda: str([x for y in self._avoid for x in y._tags]))

    @staticmethod
    def _calculate_edge_weight(src, dst, data):
        return ReachInteraction.l2_distance(
            (src._struct.player_location.x, src._struct.player_location.y),
            (dst._struct.player_location.x, dst._struct.player_location.y)
        )

    def _recalculate_target(self, arbitrary=False):
        if self._hull:
            states = [self._states[p] for p in
                        ((round(x, 2), round(y, 2)) for x, y in
                        (self._hull.points[i] for i in self._hull.vertices))]
        else:
            states = list(self._states.values())

        if arbitrary:
            self._target_state = self._entropy.choice(states)
        else:
            if all(x._cycle == self._cycles for x in states):
                self._cycles += 1
                ProfileValue('strat_cycles')(self._cycles)
            ProfileValue('strat_percent')(f'{sum(1 for x in states if x._cycle == self._cycles) / len(states) * 100:.1f}')
            self._counter = 0
            ProfileValue('strat_counter')(self._counter)
            distances = nx.shortest_path_length(self._sm._graph, self._entry, \
                                            weight=self._calculate_edge_weight, \
                                            method='bellman-ford')

            for self._target_state in sorted(
                    filter(lambda x: x in states and x._cycle < self._cycles,
                           distances.keys()),
                    key=lambda x: distances[x],
                    reverse=True
                ):
                break
            else:
                import ipdb; ipdb.set_trace()

        if self._target_state != self._preemptor._exit_state:
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
    def target(self) -> list:
        prereqs = self.target_state._prereqs.copy()

        # while True:
        #     changed = False
        #     expanded = []
        #     for p in prereqs:
        #         expanded.append(p)
        #         for sub_p in p._prereqs:
        #             if sub_p not in expanded:
        #                 expanded.append(sub_p)
        #                 changed = True
        #     prereqs = expanded
        #     if not changed:
        #         break

        for sought in self._seek:
            if sought in prereqs:
                prereqs.remove(sought)
        if not prereqs or prereqs[-1] != self.target_state:
            prereqs.append(self.target_state)
        loader = self._entry._sman._loader
        path = []
        j = None
        for i, seek in enumerate(prereqs):
            if seek in self._invalid_states:
                # `seek` might have been invalidated in the meantime, so we
                # ignore it from prereqs until further notice
                continue
            src = prereqs[j] if j is not None else None

            first = None
            attempts = 0
            for attempts, subpath in enumerate(loader.live_path_gen(
                    seek, self._entry._sman, src)):
                if attempts >= 50:
                    break
                dsts = list(map(lambda x: x[1], subpath))

                # although this is probably impossible, we add a sanity check
                # anyway
                # index = None
                # check_list = list(self._invalid_states)
                # while any(((index := i + 1), (state := x))[1] in dsts
                #         for i, x in enumerate(check_list[index:])):
                #     # this could be due to a desync between the statemanager and
                #     # the strategy; should be resolved in the next update
                #     warning(f"Invalid state found on path. Revalidating {state}")
                #     self.update_state(state)
                #     continue

                if first is None:
                    first = subpath

                if any(x in dsts for x in self._avoid):
                    continue

                # we found a path to seek the prereq without hitting any avoid
                path.extend(subpath)
                break
            if attempts >= 50 and first is not None:
                # all paths hit an avoid, so we just choose the shortest one
                path.extend(first)
            elif first is None:
                # when no valid path is found, we ignore the current prereq
                continue
            # at this point, the current prereq `seek` has been satisfied.
            # j points to the last added prereq
            j = i

        if not path:
            # if the whole path is invalid/empty, we try to recalculate the
            # target
            tmp = self.target_state
            self._recalculate_target()
            while tmp == self.target_state:
                # the target is still the same, we choose an arbitrary target
                # and hope for the best
                self._recalculate_target(arbitrary=True)
                # FIXME this can inf-loop in case of small graphs
            return self.target
        else:
            return path

    def set_state_tags(self, state: StateBase):
        if not hasattr(state, '_tags'):
            state._tags = dict()
        # add tags based on feedback struct
        if state._struct.pickup_valid:
            warning(f"Pickup {state._struct.pickup_type} valid at {state}")
            # we use the reader because the state has an offline copy of the
            # struct
            state._sman._tracker._reader.struct.pickup_valid = False
            # we also reset the offline copy in case the state gets rediscovered
            state._struct.pickup_valid = False
            if state._struct.pickup_type in self.INTERESTING_PICKUPS:
                state._tags['pickup'] = state._struct.pickup_type
        if state._struct.floor_is_lava:
            warning(f"Floor is lava at {state}")
            state._tags['lava'] = True
        if state._struct.secret_sector:
            warning(f"Secret sector at {state}")
            state._tags['secret'] = True

    def set_state_prereqs(self, state: StateBase):
        state._prereqs = self._seek.copy()

    def update_seek_avoid(self, state: StateBase):
        # we prioritize pickups (and optionally, secrets) even if they're in lava
        if 'pickup' in state._tags:
            # struct_copy = ZoomState._copy_struct(state._struct)
            # struct_copy.player_location = state._struct.pickup_location
            # adj_state = ZoomState(struct_copy)
            # if adj_state == state:
            #     # if the hashes match, then it's a cached state and we need not
            #     # copy tags and prepreqs
            #     adj_state = state
            # else:
            #     adj_state._sman = state._sman
            #     if hasattr(state, '_tags'):
            #         adj_state._tags = state._tags.copy()
            #     if hasattr(state, '_prereqs'):
            #         adj_state._prereqs = state._prereqs.copy()
            #     if hasattr(state, '_cycle'):
            #         adj_state._cycle = state._cycle
            # if adj_state in adj_state._sman._sm._graph.nodes and \
            #         adj_state not in self._seek:
            if state not in self._seek:
                self._seek.append(state)
        # elif 'secret' in state._tags:
            # if state not in self._seek:
                # self._seek.append(state)
        elif 'lava' in state._tags:
            self._avoid.add(state)

    def update_state(self, state: StateBase, invalidate: bool = False, is_new: bool = False):
        if invalidate:
            # After a state is invalidated, we might get disconnected subgraphs
            # (since we're not stitching transitions; see StateManager).
            # We need to re-adjust the target states to include only those
            # reachable from the entry state
            cull = set()
            for p, s in self._states.items():
                if not s in self._sm._graph \
                        or not nx.has_path(self._sm._graph, self._entry, s):
                    cull.add(p)
            for p in cull:
                del self._states[p]

            self.remove_point(state)
            self._invalid_states.add(state)
            if self._target_state == state:
                self._recalculate_target()
        else:
            if is_new or not hasattr(state, '_cycle'):
                # state._cycle = self._target_state._cycle = self._cycles - 1
                state._cycle = self._cycles - 1
                self.set_state_tags(state)
                self.set_state_prereqs(state)
                self.update_seek_avoid(state)
            if not nx.has_path(self._sm._graph, self._entry, state):
                # FIXME not sure when this might happen yet, but it's just a
                # preventative measure at this point
                return
            self.add_point(state)
            self._invalid_states.discard(state)

    @property
    def _points(self):
        return list(self._states.keys())

    def state_to_point(self, state):
        return (round(state._struct.player_location.x, 2),
                round(state._struct.player_location.y, 2))

    def add_point(self, state: StateBase):
        if not (point := self.state_to_point(state)) in self._states \
                and 'lava' not in state._tags:
            self._states[point] = state
            if len(self._states) >= 3:
                if not self._hull:
                    try:
                        self._hull = ConvexHull(self._points, incremental=True)
                    except QhullError as ex:
                        pass
                else:
                    self._hull.add_points([point,])

    def remove_point(self, state):
        if (point := self.state_to_point(state)) in self._states:
            del self._states[point]
        if self._hull:
            self._hull.close()
            if len(self._states) < 3:
                self._hull = None
            else:
                try:
                    self._hull = ConvexHull(self._points, incremental=True)
                except QhullError as ex:
                    self._hull = None

    def update_transition(self, source: StateBase, destination: StateBase, input: InputBase, invalidate: bool = False):
        if invalidate:
            return
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
    PICKUP_VALID = auto()
    LEVEL_FINISHED = auto()

    NUM_REASONS = auto()

class ZoomPreemptor:
    PREEMPTION_LISTS = OrderedDict([
        (0, (InterruptReason.PLAYER_DEATH,)),
        (5, (InterruptReason.LEVEL_FINISHED,)),
        (10, (InterruptReason.LAVA_PIT,)),
        (20, (InterruptReason.ATTACKER_VALID,)),
        (30, (InterruptReason.SPECIAL_OBJECT,)),
        (40, (InterruptReason.PICKUP_VALID,)),
    ])
    PREEMPTION_REPEAT_THRESHOLD = 100
    ATTACKER_MAX_TRIGGER_DISTANCE = 1000

    def __init__(self, reader, loader, entry_state, strategy):
        self._reader = reader
        self._loader = loader
        # FIXME state should not be passed around like this
        self._state = entry_state
        self._strat = strategy

        self._reason = InterruptReason.NO_REASON
        self._attacked_location = None
        self._exit_state = None
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
            if struct.player_state > 0:
                reasons.append(InterruptReason.PLAYER_DEATH)
            if struct.level_finished:
                self._exit_state = self._state._sman._tracker.current_state
                self._exit_state._cycle = -1
                struct.level_finished = False
                reasons.append(InterruptReason.LEVEL_FINISHED)
            if struct.floor_is_lava:
                reasons.append(InterruptReason.LAVA_PIT)
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
            if struct.pickup_valid:
                reasons.append(InterruptReason.PICKUP_VALID)

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
                asyncio.create_task(self.handle_reason(preemptive_reason, struct))
            elif (self._reason == InterruptReason.ATTACKER_VALID and \
                    not struct.attacker_valid) or \
                 (self._reason == InterruptReason.LEVEL_FINISHED and \
                    not struct.level_finished) or \
                 (self._reason == InterruptReason.LAVA_PIT and \
                    not struct.floor_is_lava) or \
                 (self._reason == InterruptReason.PLAYER_DEATH and \
                    struct.player_state == 0) or \
                 (self._reason == InterruptReason.SPECIAL_OBJECT and \
                    not struct.can_activate) or \
                 (self._reason == InterruptReason.PICKUP_VALID and \
                    not struct.pickup_valid):
                warning("Clearing reason, looking for new reasons")
                self._reason = InterruptReason.NO_REASON
                self._attacked_location = None

            await DelayInteraction(0.1).perform(self._loader._channel)

    async def handle_reason(self, reason, struct):
        # the struct parameter is a live reference to the shmem region
        async def internal_handle_reason():
            main_task = self._target_task
            self._target_task = asyncio.current_task()

            try:
                if self.get_reason_priority(reason) > self.get_reason_priority(self._reason):
                    warning(f"My {reason=} is less important than existing {self._reason=}. Ignoring")
                    return True

                channel = self._loader.channel
                if reason == InterruptReason.PLAYER_DEATH:
                    # FIXME is this the correct way to handle this?
                    try:
                        main_task.coro.suspend()
                        await RespawnInteraction(self._state).perform(channel)
                        await ResetKeysInteraction().perform(channel)
                        self._strat._seek.clear()
                    finally:
                        self._reason = InterruptReason.NO_REASON
                        main_task.coro.resume()
                    main_task.cancel()
                    return False
                elif reason == InterruptReason.LEVEL_FINISHED:
                    try:
                        main_task.coro.suspend()
                        await ResetKeysInteraction().perform(channel)
                        self._strat._seek.clear()
                    finally:
                        self._reason = InterruptReason.NO_REASON
                        main_task.coro.resume()
                    main_task.cancel("finished")
                    return False
                elif reason == InterruptReason.LAVA_PIT:
                    # HACK When the fuzzer is following a path using load_state,
                    # it is futile to prevent it from crossing lava. We detect
                    # that by checking if the loader is loading a state
                    if self._loader._loading:
                        warning("Skipping lava avoidance while following path")
                        return True
                    try:
                        main_task.coro.suspend()
                        sman = self._state._sman
                        current_state = sman._tracker.current_state
                        nonlava = min(filter(
                                lambda s: 'lava' not in s._tags,
                                sman.state_machine._graph.nodes), \
                            key=lambda s: ReachInteraction.l2_distance(
                                    (current_state._struct.player_location.x, current_state._struct.player_location.y),
                                    (s._struct.player_location.x, s._struct.player_location.y))
                        )
                        try:
                            await ReachInteraction(self._state,
                                (nonlava._struct.player_location.x,
                                 nonlava._struct.player_location.y),
                                sufficient_condition=lambda: not struct.floor_is_lava,
                                necessary_condition=lambda: not struct.floor_is_lava,
                                force_strafe=True).perform(channel)
                        except Exception as ex:
                            # best-effort; if we fail, whatever
                            warning(f"Failed to avoid lava {ex=}")
                    finally:
                        # self._reason = InterruptReason.NO_REASON
                        main_task.coro.resume()
                    return True
                elif reason == InterruptReason.ATTACKER_VALID:
                    try:
                        main_task.coro.suspend()
                        await KillInteraction(self._state).perform(channel)
                        self._reason = InterruptReason.NO_REASON
                    except Exception as ex:
                        warning(f"Failed to kill enemy {ex=}")
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
                elif reason == InterruptReason.PICKUP_VALID:
                    # `state` may be a cached version, so we might be
                    # influencing an existing state
                    state = ZoomState(struct)
                    state._sman = self._state._sman
                    # we reset it to true so that the strategy can handle it
                    # properly
                    state._struct.pickup_valid = True
                    self._strat.set_state_tags(state)
                    # FIXME hacky fix; when the state manager is trying to
                    # reload a target state path, it does so with
                    # update==False. The strategy state update is thus not
                    # invoked, and it is not aware that some prereqs have
                    # already been satisfied
                    self._strat.update_seek_avoid(state)
                    self._reason = InterruptReason.NO_REASON
                    return True
            except asyncio.CancelledError as ex:
                warning(f"Interrupt handler got cancelled! Propagating to {main_task=}")
                main_task.cancel(*ex.args)
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
